from openai import OpenAI
import time, json

from data import *

class BatchInference():
    def __init__(self, batch_input_file_path, batch_output_file_path, key_path, config):
        self.client = OpenAI(api_key=open(key_path,"r").read().strip())
        self.batch_input_file_path = batch_input_file_path
        self.batch_output_file_path = batch_output_file_path
        self.current_batch_instance = None
        self.current_status = None
        self.key_path = key_path
        
        if config['data_type'] == 'LaMP2':
            self.question_template = "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {text}"
        elif config['data_type'] == 'LaMP3':
            self.question_template = "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {text}"
        elif config['data_type'] == 'LaMP4':
            self.question_template = "Generate a headline for the following article: {text}"
        elif config['data_type'] == 'LaMP5':
            self.question_template = "Generate a title for the following abstract of a paper: {text}"
    
    def make_batch_zeroshot_input_data(self, dataset_iter):
        with open(self.batch_input_file_path, "w") as file:
            for row in dataset_iter:
                tmp_id = f"{row['id']}"
                tmp_input = f"{row['input']}"
                tmp_data = {
                    "custom_id": tmp_id,
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": 
                    {
                        "model": "gpt-3.5-turbo-1106",
                        "messages": [
                            {
                                "role": "user", 
                                "content": tmp_input,
                            }
                        ],
                        "max_tokens": 512,
                        "temperature": 1
                    }
                }
                file.write(json.dumps(tmp_data, ensure_ascii=False, indent=None)+"\n")

    def make_batch_adapter_input_data(self, dataset_iter):
        with open(self.batch_input_file_path, "w") as file:
            for row in dataset_iter:
                tmp_id = f"{row['id']}"
                tmp_input = f"{row['input']}"
                tmp_data = {
                    "custom_id": tmp_id,
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": 
                    {
                        "model": "gpt-3.5-turbo-1106",
                        "messages": [
                            {
                                "role": "user", 
                                "content": tmp_input,
                            }
                        ],
                        "max_tokens": 512,
                        "temperature": 1,
                        "n": 8
                    }
                }
                file.write(json.dumps(tmp_data, ensure_ascii=False, indent=None)+"\n")
                

    def call_batch_inference(self):
        batch_input_file = self.client.files.create(file=open(self.batch_input_file_path, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        self.current_batch_instance = self.client.batches.create(input_file_id=batch_input_file_id,
                                                                 endpoint="/v1/chat/completions",
                                                                 completion_window="24h")
    
    def _new_client(self):
        return OpenAI(api_key=open(self.key_path, "r").read().strip())

    def refresh_status(self):
        try:
            self.current_status = self.client.batches.retrieve(self.current_batch_instance.id).status
            print(self.current_status)
            return self.current_status
        except Exception as e:
            print(f"refresh_status error: {type(e).__name__}: {e}")
            try:
                time.sleep(60)
                self.client = self._new_client()
                self.current_status = self.client.batches.retrieve(self.current_batch_instance.id).status
                print(self.current_status)
                return self.current_status
            except Exception as e2:
                print(f"retry failed: {type(e2).__name__}: {e2}")
                raise
    
    def wait_until_done(self, delay_second=60, max_minutes=1441):
        start_time = time.time()
        while max_minutes > (time.time()-start_time)/60:
            status = self.refresh_status()
            if status in ("completed", "failed", "cancelled", "expired"):
                print(f"Batch finished with status: {status}"); return status
            time.sleep(delay_second)
        print("Timed out waiting for batch"); return None
    
    def save_batch_output_file(self):
        if self.current_status == 'completed':
            file_response = self.client.files.content(self.client.batches.retrieve(self.current_batch_instance.id).output_file_id)
            file_response.write_to_file(self.batch_output_file_path)
        else:
            print("Not Yet")
            
    def parse_output(self):
        with open(self.batch_output_file_path, "r") as file:
            parsed = [json.loads(line) for line in file.readlines()]
        output = [par["response"]["body"]["choices"][0]["message"]["content"] for par in parsed]
        return output