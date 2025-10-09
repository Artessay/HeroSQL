
if __name__ == '__main__':
    from src.meta import DBMetaAgent
    db_meta_agent = DBMetaAgent()

    database_path = 'data/ehr/mimic_iv.sqlite'
    # database_path = 'data/bird/dev/dev_databases/debit_card_specializing/debit_card_specializing.sqlite'
    
    schema = db_meta_agent.get_schema(database_path)    
    print(schema)
    print("---------------------------------------")

    schema = db_meta_agent.get_schema(database_path, tight_format=True)    
    print(schema)