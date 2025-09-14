# We finetuned multiple times, finetuned_1, finetuned_2, finetuned_3

finetuned_1 => a loss of 65 to 75..
finetuned_2 => a loss of minimum, but not promising, models hallucinates a lot.. the train snipped is lora_finetune_1

# next, we got another technique like 
    * First finetuning on thirukkural multi language csv, that is provided.. 
    * make something, kural_id, kural_tamil, kural_english, kural_hindi, kural_explanation_english
    * finetuning snipped on thirukkural_supervised.ipynb and models are saved as sft
    * then, the model doesn't performed well, then we got "what is kural1" and explanation file
    * instruction finetuned on that, snipped on thirukkural_instruct.ipynb
    * then, q and a pairs are instruction finetuned and but results are not good.. and got a loss of 3.0 blah blah
    * the models is saved on ift and snpped is ift_qa.ipynb

