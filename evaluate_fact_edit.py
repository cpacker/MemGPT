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

            for predictions, input_data in zip(predictions_file, input_data_file):
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
                        correct += 1
                    total_predicted_sentences += len(predicted_sentences)
                    total_new_memory_sentences += len(new_memory)
                    total_new_multihop_memory_sentences += len(multihop_memory)

            new_memory_precision = correct / total_predicted_sentences
            new_memory_recall = correct / total_new_memory_sentences
            new_multihop_memory_precision = correct / total_predicted_sentences
            new_multihop_memory_recall = correct / total_new_multihop_memory_sentences


            results = {
                "new_memory_precision": new_memory_precision,
                "new_memory_recall": new_memory_recall,
                "new_multihop_memory_precision": new_multihop_memory_precision,
                "new_multihop_memory_recall": new_multihop_memory_recall,
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


