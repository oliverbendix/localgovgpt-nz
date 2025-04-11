#!/bin/zsh

echo "🔍 Starting fetch and save step..."
python3 scripts/fetch_and_save_documents.py

echo "✂️  Splitting documents..."
python3 scripts/split_documents.py

echo "📤 Embedding and uploading to Pinecone..."
python3 scripts/embed_documents.py

echo "✅ Workflow complete!"
