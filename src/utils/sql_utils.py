
class SqlUtils:
    @staticmethod
    def is_result_euqal(sql_result_1, sql_result_2):
        if len(sql_result_1) != len(sql_result_2):
            return False
        
        return set(sql_result_1) == set(sql_result_2)
