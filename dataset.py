from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class PopeDataset(Dataset):
    def __init__(self, dataset, split="test", subset="default"):
        self.dataset = load_dataset(dataset, subset, split=split)
        #self.dataset = self.dataset.sort("id", reverse=True)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        if isinstance(image, str):
            image = Image.open(io.BytesIO(requests.get(img).content)).convert("RGB")

        return {
            "id": sample["id"],
            "image": image,
            "question": sample["question"],
            "category": sample["category"],
            "answer": sample.get("answer", None),
        }
