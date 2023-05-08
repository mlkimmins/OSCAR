I worked through this tutorial: https://www.youtube.com/watch?v=kCc8FmEb1nY by Andrej Karpathy to gain a deeper understanding of the model behind chatgpt. 

The code that he uses is here: https://github.com/karpathy/nanoGPT

I don't want to include those files here because it will just cause mess. if someone want's to recreate my work, then please clone the repo and use my prepare.py to prepare the dataset. And then use my config file to run the training model. 

Inorder to get reasonable training done I had to use a A100 GPU on Colab which is why you might need to cache files in a different location. 

We used data from here: https://huggingface.co/datasets/scientific_papers/viewer/pubmed/train?p=0

it contains abstracts, the article, and section information. 

We were specifically looking at the pubmed data. I don't think anyone will read this, if you are interested in the results I about that more in the powerpoint slides. The main ones are:

# Hardware

Hardware for students seems to be the biggest bottle neck since it's not possible or reasonable to run these models on conventional computers/GPUs. I also worked on a stable diffusion project for another class and this also required training on a beefy gpu.

Pretraining is expensive for these models so often we can only finetune the already tuned gpt2 weights.

Deployment of models is expensive and requires scale. requiring a gpu at all times for possible runs of the model would cost hundreds of dollars a day

# Scoring

For practical use of the fine tuned models after pretraining it doesn't really make sense to use a scoring equation. Even open ai currates the data that make chatgpt sound like a personal assistant. And even open ai uses humans to judge the quality of the output of models inorder to fine tune on the last step.

# Academic research

So much of academic research will go unread and is of low quality. Most of it should be taken out of our data.

On the other hand, so much of academic research is great and will go unread because it's too long. Fundamentally the goal of encoding long articles into useful principles is a righteous but difficult one because of the amount of manual fine tuning that would be required.

## some notes about the model

It goes abstract then article not article then abstract right now. As I mentioned verifying that the model is working properly using human examination is what open AI does, writing an entire journal article to be turned into an abstract is much more difficult than just writing toy abstracts so for now my model will generate the article given an abstract.

