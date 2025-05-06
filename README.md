# my-first-llm
PyTorch と 🤗 Transformers だけで “ゼロから” GPT-2 互換モデルを学習し、 独自データで Instruction → Output 形式の対話モデルを作る最小構成スクリプトです。 GPU が 1 枚あれば動作し、学習済み language_model.pt とトークナイザを保存できます。

---

## 📖 Overview / 概要
日本語または英語の **instruction → output** 形式コーパスを使って  
GPT-2 互換モデルを **ゼロから** 学習し、学習済みウェイト
`language_model.pt` とトークナイザを保存する最小構成スクリプトです。

特長
- **ワンファイル**：`myFirstLLM.py` だけでデータ読み込み・学習・保存・推論まで完結  
- **シンプルな依存**：PyTorch と Transformers のみ  
- **軽量**：1 枚の 8 GB GPU（例：RTX 3060）でも動作可能  
- **即テスト**：学習後すぐに `generate()` で文を生成  

---

## 🚀 Quick Start

### 1. Install Requirements
```bash
python -m venv venv && source venv/bin/activate   # optional
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece datasets
