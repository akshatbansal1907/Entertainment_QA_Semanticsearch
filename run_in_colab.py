# Google Colab runner
!pip -q install -U sentence-transformers transformers torch scikit-learn gradio

from google.colab import drive
drive.mount('/content/drive')

# If this project folder is in Drive:
%cd /content/drive/MyDrive/entertainment_qa_project
!python app.py
