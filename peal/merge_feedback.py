import json
import sys

def merge_feedback(feedback_list):
	feedback_merged = {}
	feedback_merged['selectedDataPointIndices'] = feedback_list[0]['selectedDataPointIndices']
	for feedback in feedback_list[1:]:
		feedback_merged['selectedDataPointIndices'] = list(set(feedback_merged['selectedDataPointIndices']).union(set(feedback['selectedDataPointIndices'])))

	return feedback_merged

if __name__ == "__main__":
	output_file = sys.argv[1]
	input_files = sys.argv[2:]
	feedback_list = []
	for input_file in input_files:
		feedback = json.loads(open(input_file, 'r').read())
		feedback_list.append(feedback)

	feedback_merged = merge_feedback(feedback_list)
	open(output_file, 'w').write(json.dumps(feedback_merged))