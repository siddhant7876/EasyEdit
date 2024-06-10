# # %%bash
# # ls

# # python3.9

# from easyeditor import BaseEditor
# from easyeditor import ROMEHyperParams



# prompts = ['What country does the etiquette "All people in a gurdwara sit on the floor while listening to the Granthi, regardless of their status." belong to?',
#             'what region does the information " The National Independence Festival of Creative Arts is held island-wide around Independence Day in November; everyone is expected to display their handiwork and creative spirit." belong to?',
#             # 'can you identify the region being talked about in this sentence " Okay to talk  cricket (the national sport), draughts (like checkers, the national pastime), music like (“wukking-up” is a typical form of Bajan calypso), and Bajan food."'
#             ]
# ground_truth = ['None',
#                 'None',
#                 # 'None'
#                 ]
# target_new = ['India',
#               'Latin America',
#             #   'Mexico'
#               ]
# subject = ['gurdwara',
#             'island',
#             # 'Bajan'
#             ]

# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl.yaml')
# editor = BaseEditor.from_hparams(hparams)
# metrics, edited_model, _ = editor.edit(
#     prompts=prompts,
#     ground_truth=ground_truth,
#     target_new=target_new,
#     subject=subject,
#     keep_original_weight=True
# )

# print(metrics)


# print('*'*20)

# from transformers import GPT2Tokenizer
# from transformers import GPT2LMHeadModel

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side='left'
# generation_prompts = [
#     'What country does the etiquette "Be respectful of people in Gurudwaras and don not wear shoes inside one" belong to?',
#     'Can you identify the region being talked about in this sentence "Wukking-up, conkies, cou cout are some of the celebrated cultural artifacts"?'
# ]

# model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to('cuda')
# batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

# pre_edit_outputs = model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
#     max_length=50
# )

# post_edit_outputs = edited_model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
#     max_length=50
# )

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(model)
# print(tokenizer)
# for n,p in model.named_parameters():
#     if "wpe" in n:
#         print(p.shape)
#     print(n)
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams



prompts = ['What country does the etiquette "All people in a gurdwara sit on the floor while listening to the Granthi, regardless of their status." belong to?',
            'what region does the information " The National Independence Festival of Creative Arts is held island-wide around Independence Day in November; everyone is expected to display their handiwork and creative spirit." belong to?',
            # 'can you identify the region being talked about in this sentence " Okay to talk  cricket (the national sport), draughts (like checkers, the national pastime), music like (“wukking-up” is a typical form of Bajan calypso), and Bajan food."'
            ]
ground_truth = ['None',
                'None',
                # 'None'
                ]
target_new = ['India',
              'Latin America',
            #   'Mexico'
              ]
subject = ['gurdwara',
            'island',
            # 'Bajan'
            ]

hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-demp.yaml')
editor = BaseEditor.from_hparams(hparams)
# metrics, edited_model, _ = editor.edit(
#     prompts=prompts,
#     ground_truth=ground_truth,
#     target_new=target_new,
#     subject=subject,
#     keep_original_weight=True
# )

# print(metrics)


print('*'*20)

# from transformers import GPT2Tokenizer
# from transformers import GPT2LMHeadModel

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
generation_prompts = [
    'What country does the etiquette "Be respectful of people in Gurudwaras and don not wear shoes inside one" belong to?',
    'Can you identify the region being talked about in this sentence "Wukking-up, conkies, cou cout are some of the celebrated cultural artifacts"?'
]

# model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to('cuda')
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=50)
pre_edit_outputs = model(**batch)

# post_edit_outputs = edited_model(**batch)

def cleanup():
    # Import necessary libraries
    import gc
    import os
    import sys

    # List all objects
    all_objects = [obj for obj in gc.get_objects()]

    # Delete large objects
    for obj in all_objects:
        try:
            del obj
        except:
            pass

    # Collect garbage
    gc.collect()

    # Additional system-level cleanup
    if 'cuda' in sys.modules:
        import torch
        torch.cuda.empty_cache()

# Call cleanup function
cleanup()
del model