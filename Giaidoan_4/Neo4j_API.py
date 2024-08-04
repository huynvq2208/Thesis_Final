from neo4j import GraphDatabase
import pandas as pd 

URI = "neo4j://localhost:7687"
class Neo4j_API:

    def __init__(self, URI=URI, username=None,password=None):
        self.driver = GraphDatabase.driver(URI,auth=(username,password))
    

    def run_query(self, query,params=None):
        with self.driver.session(database="neo4j") as session:
            result = session.run(query,params)
            # Convert the result to a DataFrame
            data = [record.data() for record in result]
            df = pd.DataFrame(data)
            return df if not df.empty else False

    
        

        