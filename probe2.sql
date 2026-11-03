call DBMS_AI_SERVICE.CREATE_AI_MODEL('pm','{"type":"completion","model_name":"m"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('pme','{"ai_model_name":"pm","scope":"all","url":"http://0.0.0.0:8000/api/generate","access_key":"","provider":"ollama"}');
select 'created' as step;
set @t=now(6);
call DBMS_AI_SERVICE.DROP_AI_MODEL('pm');
select 'drop_with_ep_ms' as k, timestampdiff(microsecond,@t,now(6))/1000 as ms;
