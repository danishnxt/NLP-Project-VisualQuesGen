README

__Visual Question Generation task adapted from Microsoft -> https://arxiv.org/abs/1603.06059__

Work began -> Summer 2018 (at the time code / exact technical specs for task unavailable)

* Danish Farid (currntly @ U Waterloo) - LUMS (SBASSE)
* Rabeez Riaz (currently @ McKinsey & Co.) - LUMS (SBASSE)

Work conducted: 
    
    1. Downloading dataset from -> https://www.microsoft.com/en-us/download/details.aspx?id=53670
        - Custom script used to avoid trying to download images from dead links (using http get requests)
        - 
    
    2. Data Cleaning along standard methods: 
        - Running a final check for data that had incorrect question data -> found a few rows without correct questions
        - Removing quotation marks and excess formatting
        - normalizing letter-case
        - Adding start and end tokens
        - Building a vocabulary dictionary for [word -> integer] mapping (numerical data)
        - Converting all list of strings to list of integers using vocab dictionary 
        - Converting raw (variable length) sentences into constant length slices - (length determines by max length of sentences)
            - A RNN works by having training data for a single sample at multiple time slices 
            - we went ahead and slowly "revealed" our training example sentences one token at a time, filling them 
            with Stop Tokens before that:
            
            Example dataset
            Max Length: 5 (including start token)
            
            __image: img_1.jpeg (one image)__ 
            
            Before: 
                "What is that man?"
            
            After (all tokens replaced with their integer representation):
                0/ <START> <STOP> <STOP> <STOP> <STOP> <STOP>
                1/ <START> what <STOP> <STOP> <STOP> <STOP>
                2/ <START> what is <STOP> <STOP> <STOP>
                3/ <START> what is that <STOP> <STOP>
                4/ <START> what is that man <STOP>
        

    3. Rebuilding microsoft model
        - Picecing together the VQG Microsoft model from overall information from MS paper and using Keras 
           documentation to find ways to implement features
        - Tried several methods for feeding auxiliary picture data to  
            -> including embedding data into each row of an image data
            -> 
        - Found way to include data as aux information


    4. Model definition 
        Overall model hierarchy
        
            1. Image and Sentence information 
            2. VGG-16 With last layer removed (INPUT: image) - (OUTPUT: 500 length feature vector)
            3. RNN layer (LSTM/GRU units) -> 
                (X data: Sentence pushed thru an embedding), (Y data: Next word), (Side information: Picture features)
        
        Every training sample is:
        
        -> The image is read and is run through a VGG-16 with it's last layer removed -> so every imager returns 
        a high level feature vector of (length = 500). 
        
        -> This feature vector is used as "side information" to the RNN and is the recursively modified data value over each recurrent run
        
    5. Testing
        -> Model trained on a 1080Ti for 25 epochs -> details and results in notebook
        -> Other details available in all reports

    5. Other work conducted
        -> We noticed that questions are not exactly open-ended sentences and that there must be a significant pattern to exploit (a form of questions to expect), a question starting with the word "Watermelon" is unlikely and so the model need not account for this as well. 

        A word map in the report -> {} contains a word map that shows a visualization for our questions vocabulary corpus indicating that at the first position, questions were mostly only composed to a few different words. 

        We tried to find a way to exploit this property by also providing a 1 dim "prevailance" vector to our model is auxiliary information, with the expection that this might speed up model testing and guide the optimization algo to a better minima sooner.

        We were, due to lack of availability of compute resource, not able to test this extensively but, after one or two tests, our models reached a lower loss value faster or at the same rate as they did without this "prior" vector information being passed


    6. Possible future work -> 
        -> A heavy emphasis on this 'prior' dataset information holds potential to be able to speed up models being trained on a certain type of general data. "Questions" for instance as a subset or a certain 'type' of sentence. 
