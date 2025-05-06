import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# 事前に定義されたトークナイザーをロード
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# データセットのクラスを定義
class LanguageModelingDataset(Dataset):
    def __init__(self, tokenizer, file_path='data.json'):
        self.tokenizer = tokenizer
        self.inputs = []
        self.outputs = []

        # データセットをロード
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                input_ids = tokenizer.encode(item['instruction'], add_special_tokens=True)
                output_ids = tokenizer.encode(item['output'], add_special_tokens=True)
                # 入力と出力の長さを揃える
                max_length = max(len(input_ids), len(output_ids))
                input_ids.extend([tokenizer.pad_token_id] * (max_length - len(input_ids)))
                output_ids.extend([tokenizer.pad_token_id] * (max_length - len(output_ids)))
                self.inputs.append(torch.tensor(input_ids))
                self.outputs.append(torch.tensor(output_ids))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

tokenizer.pad_token = tokenizer.eos_token
# パディング関数を定義
def collate_fn(batch):
    inputs, outputs = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    outputs_padded = pad_sequence(outputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    return inputs_padded, outputs_padded

# データセットをロード
dataset = LanguageModelingDataset(tokenizer)
# データローダーを更新
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

num_epochs = 1
# トランスフォーマーの設定を定義
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=None,
    activation_function='gelu_new',
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    summary_type='cls_index',
    summary_use_proj=True,
    summary_activation=None,
    summary_proj_to_labels=True,
    summary_first_dropout=0.1,
    scale_attn_weights=True,
    use_cache=True,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    unk_token_id=tokenizer.unk_token_id,
    num_labels=1,
    classifier_dropout=None
)

# モデルを初期化
model = GPT2LMHeadModel(config).to(device)

# オプティマイザーを定義
optimizer = AdamW(model.parameters(), lr=1e-3)
# 学習ループ
for epoch in range(num_epochs):
    for step, (inputs, outputs) in enumerate(dataloader):
        # データをモデルに入力
        outputs = model(inputs.to(device), labels=outputs.to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 進捗を表示
        if step % 100 == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

# モデルを保存
torch.save(model.state_dict(), 'language_model.pt')
tokenizer.save_pretrained('gpt2')
model.save_pretrained('language_model.pt')

# 生成用の関数
def generate(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape).to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 会話の例
print(generate("こんにちは、あなたは"))

