// cqlsh $BOLT_HOST_IP_ADDRESS -f create_data.cql
CREATE KEYSPACE mydb WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 1 };
USE mydb;
DROP TABLE satseg ;
CREATE TABLE satseg ( id UUID PRIMARY KEY,  image blob, label blob, name text, filename text );
CREATE INDEX idxLabel ON satseg (name);
CREATE INDEX idxTrain ON satseg (filename);