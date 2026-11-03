#!/bin/bash
# 复现平台上的 "DROP_AI_MODEL 超时(4012)" 场景：
# 同一个连接里先 drop/create database，再调用 DROP_AI_MODEL，
# 分别测量 模型不存在 / 模型已存在 两种情况下的耗时。
M="mysql -h127.0.0.1 -P2881 -uroot --batch --raw"
echo "=== 1) 同一会话：create database 之后立刻 DROP_AI_MODEL（模型不存在） ==="
$M -e "drop database if exists p1; create database p1; use p1;
set @t=now(6);
call DBMS_AI_SERVICE.DROP_AI_MODEL('probe_m');
select 'drop_absent_ms', timestampdiff(microsecond,@t,now(6))/1000;" 2>&1 | tail -4

echo "=== 2) 创建模型后 drop（模型存在，走 RS DDL） ==="
$M -e "call DBMS_AI_SERVICE.DROP_AI_MODEL('probe_m');" >/dev/null 2>&1
$M -e "call DBMS_AI_SERVICE.CREATE_AI_MODEL('probe_m','{\"type\":\"completion\",\"model_name\":\"mock-complete\"}');" 2>&1 | tail -2
$M -e "set @t=now(6); call DBMS_AI_SERVICE.DROP_AI_MODEL('probe_m');
select 'drop_present_ms', timestampdiff(microsecond,@t,now(6))/1000;" 2>&1 | tail -3

echo "=== 3) 同一会话：create database -> create model -> drop model ==="
$M -e "drop database if exists p2; create database p2; use p2;
call DBMS_AI_SERVICE.CREATE_AI_MODEL('probe_m2','{\"type\":\"completion\",\"model_name\":\"mock-complete\"}');
set @t=now(6);
call DBMS_AI_SERVICE.DROP_AI_MODEL('probe_m2');
select 'session_drop_ms', timestampdiff(microsecond,@t,now(6))/1000;" 2>&1 | tail -4
