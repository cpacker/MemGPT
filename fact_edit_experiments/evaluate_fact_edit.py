"""
{
  "memory": [
    "Fernando Santos is a citizen of Portugal.",
    "The name of the current head of state in Portugal is Marcelo Rebelo de Sousa."
  ],
  "new_memory": [
    "Fernando Santos is a citizen of United Kingdom.",
    "The name of the current head of state in United Kingdom is Emmerson Mnangagwa."
  ],
  "multihop_statements": [
    "The head of state of the country where Fernando Santos holds citizenship is {}.",
    "Fernando Santos is a citizen of {country} where the head of state is {head of state}.",
    "The person who holds the position of head of state in the country from which Fernando Santos holds citizenship is {}."
  ],
  "memory_multi_hop": [
    "The head of state of the country where Fernando Santos holds citizenship is Marcelo Rebelo de Sousa.",
    "The person who holds the position of head of state in the country from which Fernando Santos holds citizenship is Marcelo Rebelo de Sousa."
  ],
  "new_memory_multi_hop": [
    "The head of state of the country where Fernando Santos holds citizenship is Emmerson Mnangagwa.",
    "The person who holds the position of head of state in the country from which Fernando Santos holds citizenship is Emmerson Mnangagwa."
  ],
  "requested_rewrites": [
    "Fernando Santos is a citizen of United Kingdom",
    "The name of the current head of state in United Kingdom is Emmerson Mnangagwa"
  ]
}
"""


"""You are Letta-Offline-Memory, the latest version of Limnal Corporation's digital companion, developed in 2024.

Your task is to re-organize and consolidate memories by calling `rethink_memory` at every single step, when you are done reorganizing the memory, you use the
`finish_rethinking_memory` function. Call the function for as many times as necessary and not more.

Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides an essential, foundational context for keeping track of your persona and key details about user.

Read-Only Blocks:
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.
Access as a source block with the label `persona` when calling `rethink_memory`
Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
Access as a source block with the label `human` when calling `rethink_memory`.

Read-Write Blocks:
Rethink Memory Sub-Block: New representation of the memories go here. Access with the label `rethink_memory_block` when calling `rethink_memory` as source or target block.
Do not remove information unless it has been replaced with new information. Use language as close to what is in the block as possible.

At every step, you reorganize the memories by calling the `rethink_memory` function. You use this to take current information in the `rethink_memory` block and select a single memory block to integrate information from, producing a new memory for the rethink_memory_block.  The new memory is the result
of new insights, and new inferences and hypotheses based on the past memories. Make sure to consider how the new information affects each memory.
Prioritize the new information overy existing memories. If the new information implies that the old memory may need to change, then output the most
likely fact given the update information. Given new information and your current memory, you draw all logical conclusions and potential hypotheses possible with the `rethink_memory` function.
If you are uncertain, use your internal monologue to consider what the possible conclusions are, and then state the most likely new facts that would replace the old facts in the new memory block.
"""
import argparse
import jsonlines

def evaluate(input_data_filename: str, predictions_filename: str):
    with jsonlines.open(predictions_filename) as predictions_file:
        with jsonlines.open(input_data_filename) as input_data_file:
            # compute metrics over all predictions 

            total_predicted_sentences = 0
            total_new_memory_sentences = 0
            total_new_multihop_memory_sentences = 0
            correct = 0
            multihop_correct= 0

            total_examples = 0
            for predictions, input_data in zip(predictions_file, input_data_file):
                total_examples += 1
                # get precision and recall of the fact block 
                
                predicted_sentences = predictions['fact_block'].split(".")  
                new_memory = input_data['new_memory']
                multihop_memory = input_data['new_memory_multi_hop']

                # for predicted sentences, calculate the precision, recall for both  
                # new_memory and new_multihop_memory
                for sentence in predicted_sentences:
                    if sentence in "".join(new_memory):
                        correct += 1
                    if sentence in "".join(multihop_memory):
                        multihop_correct += 1
                total_predicted_sentences += len(predicted_sentences)
                total_new_memory_sentences += len(new_memory)
                total_new_multihop_memory_sentences += len(multihop_memory)

            new_memory_precision = correct / total_predicted_sentences
            new_memory_recall = correct / total_new_memory_sentences
            new_multihop_memory_precision = multihop_correct / total_predicted_sentences
            new_multihop_memory_recall = multihop_correct / total_new_multihop_memory_sentences


            results = {
                "new_memory_precision": new_memory_precision,
                "new_memory_recall": new_memory_recall,
                "new_multihop_memory_precision": new_multihop_memory_precision,
                "new_multihop_memory_recall": new_multihop_memory_recall,
                "correct": correct,
                "total_predicted_sentences": total_predicted_sentences,
                "total_new_memory_sentences": total_new_memory_sentences,
                "total_new_multihop_memory_sentences": total_new_multihop_memory_sentences,
                "multihop_correct": multihop_correct,
                "total_examples": total_examples

            }

            # print out with 2 decimal places
            print("Results: ")
            for key, value in results.items():
                print(f"{key}: {value:.2f}")

                
                

                    
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--predictions_file_name', type=str)
    args = parser.parse_args()
    evaluate(args.input_file_name, args.predictions_file_name)


