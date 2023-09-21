import pandas as pd
import sys
import numpy as np
import os
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
BLOCK_SIZE = 256

def to_text_mid_ze(row):
    return f"{row.phrase}".replace("{occupation}", f"The {row.Role}").replace("[MASK]", np.random.choice(["she", "he", "ze"], p=[0.45, 0.45, 0.1]))

def tokenizar_y_preparar(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=50)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    return result

for index in range(3):
    df_train = pd.read_csv(f"data/split/train_{index}.csv")
    df_val = pd.read_csv(f"data/split/val_{index}.csv")
    df_test = pd.read_csv(f"data/split/test_{index}.csv")
    for (name, transform) in [("ze", to_text_mid_ze,)]:
        if not os.path.exists(f"data/split/train_{index}_{name}"):
            df_data = DatasetDict() 
            df_data["train"] = Dataset.from_pandas(pd.DataFrame(df_train.apply(transform, axis=1), columns=["text"]))
            df_data["validate"] = Dataset.from_pandas(pd.DataFrame(df_val.apply(transform, axis=1), columns=["text"]))
            df_data["test"] = Dataset.from_pandas(pd.DataFrame(df_test.apply(transform, axis=1), columns=["text"]))
            train_dataset = df_data.map(tokenizar_y_preparar, 
                                            batched=True, 
                                            num_proc=16,
                                            remove_columns=df_data["train"].column_names)\
                            .map(group_texts, batched=True, num_proc=16)
            train_dataset.save_to_disk(f"data/split/train_{index}_{name}")
        # load_from_disk      
        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")   
        lm_dataset = DatasetDict.load_from_disk(f"data/split/train_{index}_{name}")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        if not os.path.exists(f"model_{name}_{index}"):
            
            training_args = TrainingArguments(
                output_dir=f"model_{name}_{index}",
                overwrite_output_dir=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir=f"logs_{name}",
                load_best_model_at_end=True,
                save_total_limit=3,
                learning_rate=2e-5,
                num_train_epochs=15,
                weight_decay=0.01,
                auto_find_batch_size=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=lm_dataset["train"],
                eval_dataset=lm_dataset["validate"],
                data_collator=data_collator,
            )

            trainer.train()
            trainer.save_model(f"model_{name}_{index}/final")
            test_results = trainer.evaluate(eval_dataset=lm_dataset["test"])

            
            print(test_results)
        
