from pinecone import Pinecone, ServerlessSpec
import pandas as pd


class PineconeClient:

    def __init__(self,api_key,index_name):
        self.pc = Pinecone(api_key)


        self.index_name = index_name


    def read_embedding_csv_file(self,filename):
        df = pd.read_csv(filename)
        if not df.empty:
            return df
        else:
            return None


    def get_dimension(self,df):
        """
            Dimension of vector excludes
            course_id field and user_id field
        """
        return len(df.columns) - 1


    def check_valid_name(self):
        """
            This function is used to check valid index name
            Condition:
                All letters must be lower-case.
                Index name haven't used in database.
        """
        if self.index_name.islower() and ' ' not in self.index_name:
            return True
        else:
            return False


    def check_exist_name(self,index_name):
        indexes = self.pc.list_indexes().indexes
        for index in indexes:
            if index.name == index_name:
                return True
        return False


    def create_new_index(self, index_name,dimension, metric):
        if self.check_valid_name(index_name):
            if self.check_exist_name(index_name):
                self.pc.delete_index(index_name)
            try:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                return True
            except Exception as e:
                print(e)
                return False
        else:
            return False

    def upsert_items_to_db(self, df, batch_size=500):
        vectors_to_upsert = []
        print(self.index_name)
        if self.check_exist_name(self.index_name):
            index = self.pc.Index(self.index_name)
            for i, (_, row) in enumerate(df.iterrows()):
                item_id = str(row['item_id'])
                values = row[1:].tolist()
                metadata = {'item_id': item_id}
                vectors_to_upsert.append((item_id, values, metadata))

                # When batch size is reached, upsert the current batch
                if len(vectors_to_upsert) == batch_size:
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []  # Reset the batch

            # Upsert any remaining vectors that didn't make a full batch
            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert)

            print("Vectors upserted successfully")
        else:
            print(f"Index {self.index_name} does not exist in database")


    def upsert_users_to_db(self, df, batch_size=500):
        vectors_to_upsert = []
        print(self.index_name)
        if self.check_exist_name(self.index_name):
            index = self.pc.Index(self.index_name)
            for i, (_, row) in enumerate(df.iterrows()):
                user_id = str(row['user_id'])
                values = row[1:].tolist()
                metadata = {'user_id': user_id}
                vectors_to_upsert.append((user_id, values, metadata))

                # When batch size is reached, upsert the current batch
                if len(vectors_to_upsert) == batch_size:
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []  # Reset the batch

            # Upsert any remaining vectors that didn't make a full batch
            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert)

            print("Vectors upserted successfully")
        else:
            print(f"Index {self.index_name} does not exist in database")

    def get_vector_embeddings_by_id(self,id):
        
        index = self.pc.Index(self.index_name)
        result = index.fetch(ids=[id])
        embedding = result['vectors'][id]

        
        if embedding and 'values' in embedding:
            embedding = embedding['values']

            return embedding
        else:
            return False
        
    

    def query_result(self, query_vector, k=5):
        if self.check_exist_name(self.index_name):
            index = self.pc.Index(self.index_name)
            result = index.query(vector=query_vector, top_k=k)
            if result and 'matches' in result:
                return result['matches']
        else:
            return None
    

    def push_users_to_db(self,filename):

        df = self.read_embedding_csv_file(filename)
        dimension = self.get_dimension(df)
        # self.create_new_index(index_name=self.index_name,
        #                 dimension=dimension, metric='cosine')
        self.upsert_users_to_db(df, batch_size=500)

    
    def push_items_to_db(self,filename):

        df = self.read_embedding_csv_file(filename)
        dimension = self.get_dimension(df)
        # self.create_new_index(index_name=self.index_name,
        #                 dimension=dimension, metric='cosine')
        self.upsert_items_to_db(df, batch_size=500)



import numpy as np
import pandas as pd

# embeddings = np.load('cke_embeddings.npz')

# user_embeddings = embeddings['user_embeddings']

# with open("user_list_oulad.txt", 'r') as file:
#     next(file)
#     user_list = [line.strip().split()[0] for line in file]  

# print(len(user_embeddings))
# print(len(user_list))

# if len(user_embeddings) != len(user_list):
#     raise ValueError("The number of embeddings does not match the number of users")

# # Create a DataFrame with user IDs and their corresponding embeddings
# df = pd.DataFrame({
#     'user_id': user_list,
#     'embedding': list(user_embeddings)
# })

# embedding_dim = len(user_embeddings[0])
# embedding_columns = [f'embedding_{i}' for i in range(embedding_dim)]

# df[embedding_columns] = pd.DataFrame(df['embedding'].tolist(), index=df.index)

# # Drop the original 'embedding' column if you don't need it anymore
# df = df.drop('embedding', axis=1)

# # Save the DataFrame to a CSV file
# df.to_csv('user_embeddings_mapped.csv', index=False)

# print("User embeddings have been mapped and saved to 'user_embeddings_mapped.csv'")

