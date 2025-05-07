from neo4j import GraphDatabase

def test_connection():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "Wrhgq2012838!"
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            print("Connection successful!")
            print(result.single()[0])
    except Exception as e:
        print(f"Connection failed: {str(e)}")
    finally:
        driver.close()

if __name__ == "__main__":
    test_connection() 