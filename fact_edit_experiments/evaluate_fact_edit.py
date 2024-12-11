"""
Script that generates evaluation scores given the predictions.

Example:

  python evaluate_fact_edit.py --predictions_file_name  \
      MQuAKE/datasets/letta-MQuAKE-CF-3k-v2-3000.json-predictions-offline-memory.json


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
      total_predicted_sentences = 0
      total_new_memory_sentences = 0
      total_new_multihop_memory_sentences = 0
      correct = 0
      multihop_correct = 0

      total_examples = 0
      for predictions in predictions_file:
          total_examples += 1

          predicted_sentences = predictions["final_answer"].split(". ")
          new_memory = [sentence.strip(". ") for sentence in predictions["new_memory"]]
          multihop_memory= [sentence.strip(". ") for sentence in predictions["new_memory_multi_hop"]]

          for sentence in predicted_sentences:
              if sentence in new_memory:
                  correct += 1
              if sentence in multihop_memory:
                  multihop_correct += 1
          total_predicted_sentences += len(predicted_sentences)
          total_new_memory_sentences += len(new_memory)
          total_new_multihop_memory_sentences += len(multihop_memory)

      new_memory_recall = correct / total_new_memory_sentences
      new_multihop_memory_recall = multihop_correct / total_new_multihop_memory_sentences

      results = {
          "new_memory_recall": new_memory_recall,
          "new_multihop_memory_recall": new_multihop_memory_recall,
          "correct": correct,
          "total_predicted_sentences": total_predicted_sentences,
          "total_new_memory_sentences": total_new_memory_sentences,
          "total_new_multihop_memory_sentences": total_new_multihop_memory_sentences,
          "multihop_correct": multihop_correct,
          "total_examples": total_examples,
      }

      print("Results: ")
      for key, value in results.items():
          print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", type=str)
    parser.add_argument("--predictions_file_name", type=str)
    args = parser.parse_args()
    evaluate(args.input_file_name, args.predictions_file_name)
