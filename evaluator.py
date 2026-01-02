import os
import json
from tqdm import tqdm
from corruptions import ImageCorruptor
from utils import save_comparison_plot


class PopeEvaluator:
    def __init__(self, model, dataset, output_dir, max_new_tokens, system_prompts):
        self.model = model
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        self.system_prompts = system_prompts
        self.corruptor = ImageCorruptor()

        self.gen_dir = os.path.join(output_dir, "all_generations")
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def run(self):
        for sample in tqdm(self.dataset):
            print(f"ID {sample['id']}")
            os.makedirs(f"{self.gen_dir}/{sample['id']}", exist_ok=True)
            save_path = f"{self.gen_dir}/{sample['id']}"
            images = self.corruptor.apply_all(sample["image"])
            outputs = {}

            for variant, img in images.items():
                outputs[variant] = {}
                os.makedirs(f"{save_path}/{variant}", exist_ok=True)
                img.save(f"{save_path}/{variant}/image.png")
                for i,prompt in enumerate(self.system_prompts):
                    outputs[variant][prompt] = self.model.generate_custom(
                        img,
                        sample["question"],
                        self.max_new_tokens,
                        sample["answer"],
                        prompt, 
                        f"{save_path}/{variant}/{i}"
                    )

            # Save generations
            with open(f"{self.gen_dir}/{sample['id']}/qa.json", "w") as f:
                json.dump(
                    {
                        "question": sample["question"],
                        "category": sample["category"],
                        "answer": sample["answer"],
                        "outputs": outputs,
                    },
                    f,
                    indent=2
                )

            # Save comparison plot
            save_comparison_plot(
                images,
                outputs,
                sample["question"],
                f"{save_path}/comparison.png"
            )
