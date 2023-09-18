from pathlib import Path
from dotenv import dotenv_values
from chatbert.trainer import ChatBERTTokenizerTrainer

config = dotenv_values("./.env")

tokenizer_trainer = ChatBERTTokenizerTrainer(model_path="bert-base-cased")
tokenizer_trainer.train_tokenizer(
    messages_path=Path("data/messages.json"),
    vocab_size=52000,
)
tokenizer_trainer.save_tokenizer("tokenizer")

tokenizer_trainer.push_to_hub(
    tokenizer_path="Shogunkayo/baral-bert",
    commit_message="Added tokenizer trained on WhatsApp messages",
    auth_token=config["AUTH_TOKEN"]
)
