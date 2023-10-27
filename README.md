# Chain-of-Thought-Embeddings
 
This repository contains code for the 2023 EMNLP Findings paper **Chain-of-Thought Embeddings for Stance Detection on Social Media**


In the data folder, we provide the data for Tweet-Stance along with the ChatGPT (gpt-3.5-turbo) COT outputs used in this work. The details of our prompting strategy can be found in the paper. 


### How to Run Code 

First, install requirements

```
pip install -r requirements.txt
```

Then, you can run code as follows:
```
python main.py <modality> --seed <seed> 
```

Valid args for <modality> include ```text-only, cot-only, text+cot```. We used seeds = ```[1,2,3,4,5]``` in Table 7 of our paper. 

#### Note 
We are unable to release data for presidential-stance, but you can sign a terms of use agreement and download it [here](https://portals.mdi.georgetown.edu/public/stance-detection-KE-MLM). If you are interested in more details re: our work with this dataset please reach out! joseph {dot} m {dot} gatto {dot} gr @ dartmouth {dot} edu
