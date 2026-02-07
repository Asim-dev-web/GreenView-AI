import sys, os, glob, cv2, torch, random, zipfile, shutil
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

IMG_SIZE, BATCH_SIZE = 512, 2
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

train_transform = A.Compose([
    A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE), 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.SafeRotate(limit=180, p=0.5, border_mode=cv2.BORDER_REFLECT),
    A.OneOf([A.HueSaturationValue(15, 25, 15, p=1), A.RandomBrightnessContrast(0.2, 0.2, p=1)], p=0.5),
    A.OneOf([A.GaussianBlur(3, p=1), A.GaussNoise(std_range=(0.02, 0.05), p=1)], p=0.3),
])

class UniversalTreeDataset(Dataset):
    def __init__(self, i_paths, m_paths, transform=None, is_ld=False):
        self.i_paths, self.m_paths, self.transform, self.is_ld = i_paths, m_paths, transform, is_ld
    def __len__(self): return len(self.i_paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.i_paths[idx])
        if img is None: return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.m_paths[idx], 0)
        if self.is_ld:
            mask = ((mask == 6) | (mask == 7)).astype('float32')
        else:
            mask = (mask > 0).astype('float32')
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        return torch.tensor(img.transpose(2, 0, 1)).float()/255.0, torch.tensor(mask).unsqueeze(0)

class BalancedAdaptationSampler(Sampler):
    def __init__(self, c_idx, l_idx):
        self.c_idx, self.l_idx = c_idx, l_idx
    def __iter__(self):
        random.shuffle(self.c_idx)
        sel_l = random.sample(self.l_idx, len(self.c_idx))
        combined = []
        for i in range(len(self.c_idx)):
            combined.extend([self.c_idx[i], sel_l[i]])
        return iter(combined)
    def __len__(self): return len(self.c_idx) * 2

def get_iou(p, t):
    p = (torch.sigmoid(p) > 0.5).float()
    stats = smp.metrics.get_stats(p.long(), t.long(), mode='binary')
    return smp.metrics.iou_score(*stats, reduction="micro")

def validate(model, v_loader, loss_fn, limit=None):
    model.eval()
    v_l, v_i, count = 0, 0, 0
    with torch.no_grad():
        for x, y in v_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            o = model(x); v_l += loss_fn(o, y).item(); v_i += get_iou(o, y).item()
            count += 1
            if limit and count >= limit: break
    return v_l / count, v_i / count

def run_phase(model, loader, v_ld, v_ct, loss_fn, name, ep, lr, unfreeze=False, opt=None):
    if opt is None: opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for pg in opt.param_groups: pg['lr'] = lr
    for p in model.encoder.parameters(): p.requires_grad = unfreeze
    best_v_loss = float('inf')
    for epoch in range(1, ep + 1):
        model.train()
        t_l_sum, t_i_sum = 0, 0
        pbar = tqdm(loader, desc=f"[{name}] E{epoch}/{ep}", unit="batch")
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(); o = model(x); loss = loss_fn(o, y); loss.backward(); opt.step()
            iou = get_iou(o, y).item()
            t_l_sum += loss.item(); t_i_sum += iou
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'IoU': f"{iou:.4f}"})
        l_loss, l_iou = validate(model, v_ld, loss_fn, limit=100)
        c_loss, c_iou = validate(model, v_ct, loss_fn)
        print(f"\n--- {name} E{epoch} ---")
        print(f"TRAIN      | Loss: {t_l_sum/len(loader):.4f} | IoU: {t_i_sum/len(loader):.4f}")
        print(f"VAL LoveDA | Loss: {l_loss:.4f} | IoU: {l_iou:.4f}")
        print(f"VAL Delhi  | Loss: {c_loss:.4f} | IoU: {c_iou:.4f}")
        if l_loss < best_v_loss:
            best_v_loss = l_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/{name.lower()}_best.pth")
    return opt

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"DEVICE: {torch.cuda.get_device_name(0)}")
    else:
        print("DEVICE: CPU")

    ld_i = sorted(glob.glob("LoveDA/Train/**/images/*.png", recursive=True))
    ld_m = sorted(glob.glob("LoveDA/Train/**/masks/*.png", recursive=True))
    ld_v_i = sorted(glob.glob("LoveDA/Val/**/images/*.png", recursive=True))
    ld_v_m = sorted(glob.glob("LoveDA/Val/**/masks/*.png", recursive=True))
    c_i = sorted(glob.glob("final_training_patches/images/*.jpg"))
    c_m = sorted(glob.glob("final_training_patches/masks/*.png"))

    loader_args = {'batch_size': BATCH_SIZE, 'num_workers': 4, 'pin_memory': True}
    ds_ld = UniversalTreeDataset(ld_i, ld_m, train_transform, True)
    v_ld = DataLoader(UniversalTreeDataset(ld_v_i, ld_v_m, None, True), **loader_args)
    idx = list(range(len(c_i))); random.shuffle(idx); t_idx, v_idx = idx[8:], idx[:8]
    ds_ct = UniversalTreeDataset([c_i[k] for k in t_idx], [c_m[k] for k in t_idx], train_transform, False)
    v_ct = DataLoader(UniversalTreeDataset([c_i[k] for k in v_idx], [c_m[k] for k in v_idx], None, False), **loader_args)

    model = smp.UnetPlusPlus("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    loss_fn = lambda p, t: 2.0 * smp.losses.TverskyLoss(mode='binary', alpha=0.5, beta=0.5, from_logits=True)(p, t) + 0.5 * torch.nn.BCEWithLogitsLoss()(p, t)

    opt = run_phase(model, DataLoader(ds_ld, shuffle=True, **loader_args), v_ld, v_ct, loss_fn, "Warmup", 5, 1e-4)
    opt = run_phase(model, DataLoader(ds_ld, shuffle=True, **loader_args), v_ld, v_ct, loss_fn, "General", 15, 5e-5, True, opt)
    sampler = BalancedAdaptationSampler(list(range(len(ds_ct))), list(range(len(ds_ct), len(ds_ct)+len(ds_ld))))
    run_phase(model, DataLoader(ConcatDataset([ds_ct, ds_ld]), sampler=sampler, **loader_args), v_ld, v_ct, loss_fn, "Adapt", 15, 1e-5, True, opt)
    torch.save(model.state_dict(), "final_model.pth")