See main.ipynb for the jupyter notebook implementing the assignment. 

I implemented CASE 1, using the cosine distance to compute the distance between consecutive slices. I set a fixed context window size of 8192 bytes, to comfortably stay within the Llama-13b's (the model that I'm using) context window limit of 4096 tokens (much smaller than the 128MB limit indicated in the assignment; however, I'm using a private api key which cannot, for rate limit reasons, handle documents that large).

Document.py implements a helper Document class that is used for text documents. It stores, internally, a Bag of Word representation. It also implements the cosine_distance function.

