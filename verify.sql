-- End-to-end checks for the AI function work (run against the mock AI service)
-- obclient -h127.0.0.1 -P2881 -uroot < /tmp/verify.sql

-- ---------------------------------------------------------------- endpoints
call DBMS_AI_SERVICE.CREATE_AI_MODEL('v_complete', '{"type":"completion","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e_complete',
  '{"ai_model_name":"v_complete","url":"http://127.0.0.1:8000/v1/chat/completions","access_key":"sk-mock","provider":"openai"}');

call DBMS_AI_SERVICE.CREATE_AI_MODEL('v_embed', '{"type":"dense_embedding","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e_embed',
  '{"ai_model_name":"v_embed","url":"http://127.0.0.1:8000/v1/embeddings","access_key":"sk-mock","provider":"openai"}');

call DBMS_AI_SERVICE.CREATE_AI_MODEL('v_rerank', '{"type":"rerank","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e_rerank',
  '{"ai_model_name":"v_rerank","url":"http://127.0.0.1:8000/v1/rerank","access_key":"sk-mock","provider":"siliconflow"}');

-- ollama, no access key at all
call DBMS_AI_SERVICE.CREATE_AI_MODEL('v_ollama_c', '{"type":"completion","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e_ollama_c',
  '{"ai_model_name":"v_ollama_c","url":"http://127.0.0.1:8000/api/generate","provider":"OLLAMA"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL('v_ollama_e', '{"type":"dense_embedding","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e_ollama_e',
  '{"ai_model_name":"v_ollama_e","url":"http://127.0.0.1:8000/api/embed","provider":"ollama"}');

-- ------------------------------------------------------------ 1. ai_prompt
select ai_prompt('{0}+{1}={2} and again {2}+{1}={0}', '1', '2', '3') as p_repeat;
select ai_prompt('{0}', json_object('k', json_array(1, 2))) as p_json_obj;
select ai_prompt('{0}-{1}', json_array('a', 2), json_object('b', true)) as p_json_args;
select ai_prompt('中文模板{0}，带转义\n\t{1}', '世界', 'tab\\"quote') as p_unicode;
select ai_prompt('{0}', ai_prompt('inner {0}', 'x')) as p_nested;

create table t_prompt(tmpl text, val longtext);
insert into t_prompt values('hello {0}!', '世界');
insert into t_prompt values('long {0}', repeat('x', 300));
select ai_prompt(tmpl, val) from t_prompt;

-- ------------------------------------------------------------- 2. ai_embed
select ai_embed('v_embed', 'hello') as e_default;
select ai_embed('v_embed', 'hello', 8) as e_dim8;
select json_length(ai_embed('v_embed', 'hello')) as e_len;

-- ---------------------------------------------------------- 3. ai_complete
select ai_complete('v_complete', 'hello') as c_basic;
select ai_complete('v_complete', 'hello', '{"temperature":0.5,"max_tokens":16}') as c_config;
select ai_complete('v_complete', ai_prompt('say {0}', 'hi')) as c_prompt;
select ai_complete('v_complete', repeat('long ', 200)) as c_longtext;

-- ------------------------------------------------------------ 4. ollama
select ai_complete('v_ollama_c', 'hello') as o_complete;
select ai_embed('v_ollama_e', 'hello') as o_embed;
select ai_embed('v_ollama_e', 'hello', 6) as o_embed_dim;

-- ------------------------------------------------------------ 5. ai_rerank
select ai_rerank('v_rerank', 'q', json_array('a', 'bbbb', 'cc')) as r_strings;
select ai_rerank('v_rerank', 'q',
  json_array(json_object('id', 1, 'txt', 'a'),
             json_object('id', 2, 'txt', 'bbbb'),
             json_object('id', 3, 'txt', 'cc')), 'txt') as r_objects;

-- ------------------------------------------------------------ 6. batch embed
create table t_batch(id int, c varchar(100));
insert into t_batch values(1, 'alpha'), (2, 'bravo'), (3, null), (4, 'delta'), (5, 'echo');
select id, ai_embed('v_embed', c) from t_batch order by id;
select id, ai_embed('v_embed', c) from t_batch where id > 2 order by id;
select id, ai_embed('v_embed', c) from t_batch order by id limit 2;

drop table t_prompt;
drop table t_batch;
