# -*- coding: utf-8 -*-


# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from flask import Flask, request, jsonify, render_template
import logging as log
import json
import datetime
import urllib
import re
import time
import textwrap
import pandas as pd
from flask_cors import CORS
from dbconnectors import pgconnector, bqconnector
from embeddings.store_embeddings import add_sql_embedding
from agents import EmbedderAgent, BuildSQLAgent, DebugSQLAgent, ValidateSQLAgent, VisualizeAgent, ResponseAgent

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)


import configparser
config = configparser.ConfigParser()
config.read('config.ini')

PROJECT_ID = config['GCP']['PROJECT_ID']
DATA_SOURCE = config['CONFIG']['DATA_SOURCE']
VECTOR_STORE = config['CONFIG']['VECTOR_STORE']
PG_SCHEMA = config['PGCLOUDSQL']['PG_SCHEMA']
PG_DATABASE = config['PGCLOUDSQL']['PG_DATABASE']
PG_USER = config['PGCLOUDSQL']['PG_USER']
PG_REGION = config['PGCLOUDSQL']['PG_REGION'] 
PG_INSTANCE = config['PGCLOUDSQL']['PG_INSTANCE'] 
PG_PASSWORD = config['PGCLOUDSQL']['PG_PASSWORD']
BQ_OPENDATAQNA_DATASET_NAME = config['BIGQUERY']['BQ_OPENDATAQNA_DATASET_NAME']
BQ_LOG_TABLE_NAME = config['BIGQUERY']['BQ_LOG_TABLE_NAME'] 
BQ_DATASET_REGION = config['BIGQUERY']['BQ_DATASET_REGION']
BQ_DATASET_NAME = config['BIGQUERY']['BQ_DATASET_NAME']
BQ_TABLE_LIST = config['BIGQUERY']['BQ_TABLE_LIST']


embedder = EmbedderAgent('vertex') 

SQLBuilder = BuildSQLAgent('gemini-1.0-pro')
SQLChecker = ValidateSQLAgent('gemini-1.0-pro')
SQLDebugger = DebugSQLAgent('gemini-1.0-pro')
Responder = ResponseAgent('gemini-1.0-pro')
Visualize = VisualizeAgent ()

num_table_matches = 5
num_column_matches = 10
similarity_threshold = 0.3
num_sql_matches=3

if DATA_SOURCE=='bigquery':
    USER_DATABASE=BQ_DATASET_NAME 
    src_connector = bqconnector
else: 
    USER_DATABASE=PG_SCHEMA
    src_connector = pgconnector

# Set the vector store paramaters
if VECTOR_STORE=='bigquery-vector':
    instance_name=None
    database_name=BQ_OPENDATAQNA_DATASET_NAME
    database_user=None
    database_password=None
    region=BQ_DATASET_REGION
    vector_connector = bqconnector
    call_await = False

else:
    instance_name=PG_INSTANCE
    database_name=PG_DATABASE
    database_user=PG_USER
    database_password=PG_PASSWORD
    region=PG_REGION
    vector_connector = pgconnector
    call_await=True

RUN_DEBUGGER = True 
EXECUTE_FINAL_SQL = True 

app = Flask(__name__) 
cors = CORS(app, resources={r"/*": {"origins": "*"}})



@app.route("/available_databases", methods=["GET"])
def getBDList():
    try:

        # final_sql_bq="""SELECT DISTINCT concat(table_schema, '-bigquery') as table_schema from table_embeddings"""
        # result_bq_df = pgconnector.retrieve_df(final_sql_bq)

#         final_sql="""SELECT
#   DISTINCT CONCAT(table_schema, (CASE
#         WHEN table_schema IN ('HHS_Program_Counts','fda_food') THEN '-bigquery'
#       ELSE
#       '-postgres'
#     END
#       )) AS table_schema
# FROM
#   table_details_embeddings"""
        final_sql="""SELECT
  DISTINCT CONCAT(table_schema,'-',source_type) AS table_schema
FROM
  table_details_embeddings"""
        result = pgconnector.retrieve_df(final_sql)
    
        print(result)
    
        responseDict = { 
                "ResponseCode" : 200, 
                "KnownDB" : result.to_json(orient='records'),
                "Error":""
                } 
        return jsonify(responseDict)
    except Exception as e:
        # util.write_log_entry("Issue was encountered while generating the SQL, please check the logs!" + str(e))
        responseDict = { 
                "ResponseCode" : 500, 
                "KnownDB" : "There was problem finding connecting to the resource. Please retry later!.",
                "Error":"Issue was encountered while generating the SQL, please check the logs!"  + str(e)
                } 
        return jsonify(responseDict)




@app.route("/embed_sql", methods=["POST"])
async def embedSql():
    try:
        envelope = str(request.data.decode('utf-8'))
        envelope=json.loads(envelope)
        user_database=envelope.get('user_database')
        final_sql = envelope.get('generated_sql')
        user_question = envelope.get('user_question')
            
        sql_embedding_df = await add_sql_embedding(user_question, final_sql,user_database)
        responseDict = { 
                   "ResponseCode" : 201, 
                   "Message" : "Example SQL has been accepted for embedding",
                   "Error":""
                   } 
        return jsonify(responseDict)
    except Exception as e:
        # util.write_log_entry("Issue was encountered while generating the SQL, please check the logs!" + str(e))
        responseDict = { 
                   "ResponseCode" : 500, 
                   "KnownDB" : "There was problem finding connecting to the resource. Please retry later!.",
                   "Error":"Issue was encountered while embedding the SQL as example."  + str(e)
                   } 
        return jsonify(responseDict)




@app.route("/run_query", methods=["POST"])
def getSQLResult():
    try:
        envelope = str(request.data.decode('utf-8'))
        envelope=json.loads(envelope)
        print("request payload: "+ str(envelope))
        user_database = envelope.get('user_database')
        final_sql = envelope.get('generated_sql')

        DATA_SOURCE = get_source_type(user_database)

        if DATA_SOURCE=='bigquery':
            result = bqconnector.retrieve_df(final_sql)
        else:
            result = pgconnector.retrieve_df(final_sql)
        df_len = result[result.columns[0]].count()
        if df_len > 10:
            result = result.head(10)
        print(result.to_json(orient='records'))

        responseDict = { 
                   "ResponseCode" : 200, 
                   "KnownDB" : result.to_json(orient='records'),
                   "Error":""
                   } 
        return jsonify(responseDict)
    except Exception as e:
        # util.write_log_entry("Issue was encountered while running the generated SQL, please check the logs!" + str(e))
        responseDict = { 
                   "ResponseCode" : 500, 
                   "KnownDB" : "",
                   "Error":"Issue was encountered while running the generated SQL, please check the logs!" + str(e)
                   } 
        return jsonify(responseDict)




@app.route("/get_known_sql", methods=["POST"])
def getKnownSQL():
    print("Extracting the known SQLs from the example embeddings.")
    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)
    
    user_database = envelope.get('user_database')


    try:

        final_sql="""select distinct
        example_user_question,
        example_generated_sql 
        from example_prompt_sql_embeddings
        where table_schema = '{user_database}' LIMIT 5""".format(user_database=user_database)
        print("Executing SQL: "+ final_sql)
        result = pgconnector.retrieve_df(final_sql)
        print(result.to_json(orient='records'))

        responseDict = { 
                   "ResponseCode" : 200, 
                   "KnownSQL" : result.to_json(orient='records'),
                   "Error":""
                   } 
        return jsonify(responseDict)
    except Exception as e:
        # util.write_log_entry("Issue was encountered while generating the SQL, please check the logs!" + str(e))
        responseDict = { 
                   "ResponseCode" : 500, 
                   "KnownSQL" : "There was problem finding connecting to the resource. Please retry later!.",
                   "Error":"Issue was encountered while generating the SQL, please check the logs!"  + str(e)
                   } 
        return jsonify(responseDict)


def get_source_type(user_database):
    sql=f'''SELECT
  DISTINCT source_type AS table_schema
FROM
  table_details_embeddings where table_schema='{user_database}' '''
    result = pgconnector.retrieve_df(sql)
    
    print("Source Found: "+ str(result.iloc[0, 0]))
    # if user_database in ('HHS_Program_Counts','fda_food'):
    #     return "bigquery"
    # else:
    #     return "cloudsql-pg"
    return str(result.iloc[0, 0])

@app.route("/generate_sql", methods=["POST"])
async def generateSQL():
   AUDIT_TEXT=''
   found_in_vector = 'N'
   process_step=''
   final_sql='Not Generated Yet'
   envelope = str(request.data.decode('utf-8'))
#    print("Here is the request payload " + envelope)
   envelope=json.loads(envelope)
   
   user_question = envelope.get('user_question')
   user_database = envelope.get('user_database')
   corrected_sql = ''
   DATA_SOURCE = get_source_type(user_database)
#    print(DATA_SOURCE)
   
   try:
        # Fetch the embedding of the user's input question 
    embedded_question = embedder.create(user_question)

    # Reset AUDIT_TEXT
    AUDIT_TEXT = ''

    AUDIT_TEXT = AUDIT_TEXT + "\nUser Question : " + str(user_question) + "\nUser Database : " + str(USER_DATABASE)
    process_step = "\n\nGet Exact Match: "
    # Look for exact matches in known questions 
    exact_sql_history = vector_connector.getExactMatches(user_question.replace("'","''"))

    if exact_sql_history is not None:
        found_in_vector = 'Y' 
        final_sql = exact_sql_history
        invalid_response = False
        AUDIT_TEXT = AUDIT_TEXT + "\nExact match has been found! Going to retreive the SQL query from cache and serve!" 



    else:
        # No exact match found. Proceed looking for similar entries in db 
        AUDIT_TEXT = AUDIT_TEXT +  process_step + "\nNo exact match found in query cache, retreiving revelant schema and known good queries for few shot examples using similarity search...."
        process_step = "\n\nGet Similar Match: "
        if call_await:
            similar_sql = await vector_connector.getSimilarMatches('example', USER_DATABASE, embedded_question, num_sql_matches, similarity_threshold)
        else:
            similar_sql = vector_connector.getSimilarMatches('example', USER_DATABASE, embedded_question, num_sql_matches, similarity_threshold)

        process_step = "\n\nGet Table and Column Schema: "
        # Retrieve matching tables and columns
        if call_await: 
            table_matches =  await vector_connector.getSimilarMatches('table', USER_DATABASE, embedded_question, num_table_matches, similarity_threshold)
            column_matches =  await vector_connector.getSimilarMatches('column', USER_DATABASE, embedded_question, num_column_matches, similarity_threshold)
        else:
            table_matches =  vector_connector.getSimilarMatches('table', USER_DATABASE, embedded_question, num_table_matches, similarity_threshold)
            column_matches =  vector_connector.getSimilarMatches('column', USER_DATABASE, embedded_question, num_column_matches, similarity_threshold)

        AUDIT_TEXT = AUDIT_TEXT +  process_step + "\nRetrieved Similar Known Good Queries, Table Schema and Column Schema: \n" + '\nRetrieved Tables: \n' + str(table_matches) + '\n\nRetrieved Columns: \n' + str(column_matches) + '\n\nRetrieved Known Good Queries: \n' + str(similar_sql)
        # If similar table and column schemas found: 
        if len(table_matches.replace('Schema(values):','').replace(' ','')) > 0 or len(column_matches.replace('Column name(type):','').replace(' ','')) > 0 :

            # GENERATE SQL
            process_step = "\n\nBuild SQL: "
            generated_sql = SQLBuilder.build_sql(DATA_SOURCE,user_question,table_matches,column_matches,similar_sql)
            final_sql=generated_sql
            AUDIT_TEXT = AUDIT_TEXT + process_step +  "\nGenerated SQL: " + str(generated_sql)
            
            if 'unrelated_answer' in generated_sql :
                invalid_response=True

            # If agent assessment is valid, proceed with checks  
            else:
                invalid_response=False

                if RUN_DEBUGGER: 
                    generated_sql, invalid_response, AUDIT_TEXT = SQLDebugger.start_debugger(DATA_SOURCE, generated_sql, user_question, SQLChecker, table_matches, column_matches, AUDIT_TEXT, similar_sql) 
                    # AUDIT_TEXT = AUDIT_TEXT + '\n Feedback from Debugger: \n' + feedback_text

                final_sql=generated_sql
                AUDIT_TEXT = AUDIT_TEXT + "\nFinal SQL after Debugger: \n" +str(final_sql)


        # No matching table found 
        else:
            invalid_response=True
            print('No tables found in Vector ...')
            AUDIT_TEXT = AUDIT_TEXT + "\nNo tables have been found in the Vector DB. The question cannot be answered with the provide data source!"

    
    if not invalid_response:
        responseDict = { 
                "ResponseCode" : 200, 
                "GeneratedSQL" : final_sql,
                "Error":""
                }          
        
        # return jsonify(responseDict)

    else:  # Do not execute final SQL

        print("Not executing final SQL as it is invalid, please debug!")
        response = "I am sorry, I could not come up with a valid SQL."
        _resp = Responder.run(user_question, response)
        # print(_resp)
        AUDIT_TEXT = AUDIT_TEXT + "\nModel says " + str(_resp) 

    

        responseDict = { 
                   "ResponseCode" : 200, 
                   "GeneratedSQL" : _resp,
                   "Error":""
                   }

    bqconnector.make_audit_entry(DATA_SOURCE, user_database, "gemini-1.0-pro", user_question, final_sql, found_in_vector, "", process_step, "", AUDIT_TEXT)
    
    return jsonify(responseDict)

   except Exception as e:
    # util.write_log_entry("Issue was encountered while generating the SQL, please check the logs!" + str(e))
    responseDict = { 
                   "ResponseCode" : 500, 
                   "GeneratedSQL" : "",
                   "Error":"Issue was encountered while generating the SQL, please check the logs! "  + str(e)
                   } 
    # Make Audit entry into BQ
    bqconnector.make_audit_entry(DATA_SOURCE, user_database, "gemini-1.0-pro", user_question, final_sql, found_in_vector, "", process_step, str(e), AUDIT_TEXT)
    
    return jsonify(responseDict)

@app.route("/generate_viz", methods=["POST"])
async def generateViz():
    envelope = str(request.data.decode('utf-8'))
    # print("Here is the request payload " + envelope)
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    generated_sql = envelope.get('generated_sql')
    sql_results = envelope.get('sql_results')

    chart_js=''

    try:
        chart_js = Visualize.generate_charts(user_question,generated_sql,sql_results)
        responseDict = { 
        "ResponseCode" : 200, 
        "GeneratedChartjs" : chart_js,
        "Error":""
        }
        return jsonify(responseDict)

    except Exception as e:
        # util.write_log_entry("Cannot generate the Visualization!!!, please check the logs!" + str(e))
        responseDict = { 
                "ResponseCode" : 500, 
                "GeneratedSQL" : "",
                "Error":"Issue was encountered while generating the Google Chart, please check the logs!"  + str(e)
                } 
        return jsonify(responseDict)

@app.route("/summarize_results", methods=["POST"])
async def getSummary():
    AUDIT_TEXT='Creating Summary '
    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)
   
    user_question = envelope.get('user_question')
    sql_results = envelope.get('sql_results')
    
    try:
        summary_response = Responder.run(user_question,sql_results)
        if summary_response:
            responseDict = { 
                    "ResponseCode" : 200, 
                    "summary_response" : summary_response,
                    "Error":""
                    } 
        else:
              
                AUDIT_TEXT= AUDIT_TEXT + '\n Cannot generate the Summarization! \n'
                responseDict = { 
                    "ResponseCode" : 500, 
                    "summary_response" : summary_response,
                    "Error":"Oopss!!! Cannot generate the summary! !!!!!"
                    }            
        print(AUDIT_TEXT)
        return jsonify(responseDict)
    except Exception as e:
        AUDIT_TEXT=AUDIT_TEXT+ "Cannot generate the Summarization!!!, please check the logs!" + str(e)
        responseDict = { 
                    "ResponseCode" : 500, 
                    "summary_response" : "",
                    "Error":"Issue was encountered while summarizing the results, please check the logs!"  + str(e)
                    }
        print(AUDIT_TEXT)
        return jsonify(responseDict)




@app.route("/natural_response", methods=["POST"])
async def getNaturalResponse():
   AUDIT_TEXT=''
   found_in_vector = 'N'
   process_step=''
   final_sql='Not Generated Yet'
   envelope = str(request.data.decode('utf-8'))
   #print("Here is the request payload " + envelope)
   envelope=json.loads(envelope)
   
   user_question = envelope.get('user_question')
   user_database = envelope.get('user_database')
   corrected_sql = ''
   DATA_SOURCE = get_source_type(user_database)
   AUDIT_TEXT = AUDIT_TEXT + "User Question : " + str(user_question) + "\nUser Database : " + str(user_database) + "\nSource : " + DATA_SOURCE
   try: 
        # Fetch the embedding of the user's input question 
    embedded_question = embedder.create(user_question)

    # Reset AUDIT_TEXT
    AUDIT_TEXT = ''

    AUDIT_TEXT = AUDIT_TEXT + "\nUser Question : " + str(user_question) + "\nUser Database : " + str(USER_DATABASE)
    process_step = "\n\nGet Exact Match: "
    # Look for exact matches in known questions 
    exact_sql_history = vector_connector.getExactMatches(user_question) 

    if exact_sql_history is not None:
        found_in_vector = 'Y' 
        final_sql = exact_sql_history
        invalid_response = False
        AUDIT_TEXT = AUDIT_TEXT + "\nExact match has been found" 


    else:
        # No exact match found. Proceed looking for similar entries in db 
        AUDIT_TEXT = AUDIT_TEXT +  process_step + "\nNo exact match found looking for similar entries and schema"
        process_step = "\n\nGet Similar Match: "
        if call_await:
            similar_sql = await vector_connector.getSimilarMatches('example', USER_DATABASE, embedded_question, num_sql_matches, similarity_threshold)
        else:
            similar_sql = vector_connector.getSimilarMatches('example', USER_DATABASE, embedded_question, num_sql_matches, similarity_threshold)

        process_step = "\n\nGet Table and Column Schema: "
        # Retrieve matching tables and columns
        if call_await: 
            table_matches =  await vector_connector.getSimilarMatches('table', USER_DATABASE, embedded_question, num_table_matches, similarity_threshold)
            column_matches =  await vector_connector.getSimilarMatches('column', USER_DATABASE, embedded_question, num_column_matches, similarity_threshold)
        else:
            table_matches =  vector_connector.getSimilarMatches('table', USER_DATABASE, embedded_question, num_table_matches, similarity_threshold)
            column_matches =  vector_connector.getSimilarMatches('column', USER_DATABASE, embedded_question, num_column_matches, similarity_threshold)

        AUDIT_TEXT = AUDIT_TEXT +  process_step + "\nRetrieved Similar Entries, Table Schema and Column Schema: \n" + '\nRetrieved Tables: \n' + str(table_matches) + '\n\nRetrieved Columns: \n' + str(column_matches) + '\n\nRetrieved Known Good Queries: \n' + str(similar_sql)
        # If similar table and column schemas found: 
        if len(table_matches.replace('Schema(values):','').replace(' ','')) > 0 or len(column_matches.replace('Column name(type):','').replace(' ','')) > 0 :

            # GENERATE SQL
            process_step = "\n\nBuild SQL: "
            generated_sql = SQLBuilder.build_sql(DATA_SOURCE,user_question,table_matches,column_matches,similar_sql)
            final_sql=generated_sql
            AUDIT_TEXT = AUDIT_TEXT + process_step +  "\nGenerated SQL: " + str(generated_sql)
            
            if 'unrelated_answer' in generated_sql :
                invalid_response=True

            # If agent assessment is valid, proceed with checks  
            else:
                invalid_response=False

                if RUN_DEBUGGER: 
                    generated_sql, invalid_response, AUDIT_TEXT = SQLDebugger.start_debugger(DATA_SOURCE, generated_sql, user_question, SQLChecker, table_matches, column_matches, AUDIT_TEXT, similar_sql) 
                    # AUDIT_TEXT = AUDIT_TEXT + '\n Feedback from Debugger: \n' + feedback_text

                final_sql=generated_sql
                AUDIT_TEXT = AUDIT_TEXT + "\nFinal SQL after Debugger \n: " +str(final_sql)


        # No matching table found 
        else:
            invalid_response=True
            print('No tables found in Vector ...')
            AUDIT_TEXT = AUDIT_TEXT + "\n No tables have been found in the Vector DB..."


    if not invalid_response:
        try:
            process_step="Execute the SQL"
            if EXECUTE_FINAL_SQL is True:

                if DATA_SOURCE=='bigquery':
                    result = bqconnector.retrieve_df(final_sql)
                else:
                    result = pgconnector.retrieve_df(final_sql)
                # df_len = result[result.columns[0]].count()
                # if df_len > 10:
                #     result = result.head(10)
                # print(result.to_json(orient='records'))

                try:
                    process_step="Generate Summary"
                    summary_response = Responder.run(user_question,result)
                    AUDIT_TEXT = AUDIT_TEXT + "\n Model says " + str(summary_response) 
                    if summary_response:
                        responseDict = { 
                                "ResponseCode" : 200, 
                                "summary_response" : summary_response,
                                "Error":""
                                }
                except Exception as e:
                    AUDIT_TEXT = AUDIT_TEXT + "\n Model couldn't provided natural response. Error:  " +  str(e) 
                    responseDict = { 
                                "ResponseCode" : 500, 
                                "summary_response" : "",
                                "Error":"Issue was encountered while summarizing the results, please check the logs!"  + str(e)
                                } 
                return jsonify(responseDict) 
            else:
                print("Not executing final SQL since EXECUTE_FINAL_SQL variable is False\n ")
                response = "Please enable the Execution of the final SQL so I can provide an answer"
                summary_response=Responder.run(user_question, response)
                AUDIT_TEXT = AUDIT_TEXT + "\n Model says " + str(summary_response) 
                responseDict = { 
                        "ResponseCode" : 500, 
                        "summary_response" : summary_response,
                        "Error":"Oopss!!! Cannot generate the summary! !!!!!"
                        }            

            return jsonify(responseDict)
            

        except Exception as e:
            AUDIT_TEXT = AUDIT_TEXT + "Error while generating SQL " + str(e) 
            responseDict = { 
                    "ResponseCode" : 500, 
                    "KnownDB" : "",
                    "Error":"Issue was encountered while running the generated SQL, please check the logs!" + str(e)
                    } 
            return jsonify(responseDict)

    else:  # Do not execute final SQL
        print("Not executing final SQL as it is invalid, please debug!")
        response = "I am sorry, I could not come up with a valid SQL."
        _resp = Responder.run(user_question, response)
        # print(_resp)
        AUDIT_TEXT = AUDIT_TEXT + "\n Model says " + str(_resp) 
        responseDict = { 
                "ResponseCode" : 200, 
                "GeneratedSQL" : _resp,
                "Error":""
                }
    bqconnector.make_audit_entry(DATA_SOURCE, user_database, "gemini-1.0-pro", user_question, final_sql, found_in_vector, "", process_step, "", AUDIT_TEXT)
    return jsonify(responseDict)

   except Exception as e:
        # util.write_log_entry("Issue was encountered while generating the SQL, please check the logs!" + str(e))
        responseDict = { 
                    "ResponseCode" : 500, 
                    "summary_response" : "",
                    "Error":"Issue was encountered while generating the SQL, please check the logs!"  + str(e)
                    } 
        return jsonify(responseDict)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))