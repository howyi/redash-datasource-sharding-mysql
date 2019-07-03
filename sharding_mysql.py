# -*- coding: utf-8 -*-

import logging
import os
import numpy as np

from redash.query_runner import *
from redash.utils import json_dumps, json_loads
from redash.settings import parse_boolean
from redash.query_runner.mysql import Mysql

logger = logging.getLogger(__name__)
types_map = {
    0: TYPE_FLOAT,
    1: TYPE_INTEGER,
    2: TYPE_INTEGER,
    3: TYPE_INTEGER,
    4: TYPE_FLOAT,
    5: TYPE_FLOAT,
    7: TYPE_DATETIME,
    8: TYPE_INTEGER,
    9: TYPE_INTEGER,
    10: TYPE_DATE,
    12: TYPE_DATETIME,
    15: TYPE_STRING,
    16: TYPE_INTEGER,
    246: TYPE_FLOAT,
    253: TYPE_STRING,
    254: TYPE_STRING,
}


class ShardingMysql(Mysql):
    @classmethod
    def name(cls):
        return "MySQL (Sharding)"

    @classmethod
    def type(cls):
        return 'sharding_mysql'

    @classmethod
    def enabled(cls):
        return True

    @classmethod
    def annotate_query(cls):
        return True

    @classmethod
    def configuration_schema(cls):
        show_ssl_settings = parse_boolean(os.environ.get('MYSQL_SHOW_SSL_SETTINGS', 'true'))

        schema = {
            'type': 'object',
            'properties': {
                'params': {
                    'type': 'string',
                    'default': 'shard1, shard2, shard3',
                    'title': 'Shard Parameter (Replace on {param})'
                },
                'show_params': {
                    'type': 'boolean',
                    'title': 'Display database names in the first column'
                },
                'host': {
                    'type': 'string',
                    'default': '127.0.0.1'
                },
                'user': {
                    'type': 'string'
                },
                'passwd': {
                    'type': 'string',
                    'title': 'Password'
                },
                'db': {
                    'type': 'string',
                    'title': 'Database name',
                    'default': 'test_{param}'
                },
                'port': {
                    'type': 'string',
                    'default': '3306',
                }
            },
            "order": ['params', 'show_params', 'host', 'port', 'user', 'passwd', 'db'],
            'required': ['db'],
            'secret': ['passwd']
        }

        if show_ssl_settings:
            schema['properties'].update({
                'use_ssl': {
                    'type': 'boolean',
                    'title': 'Use SSL'
                },
                'ssl_cacert': {
                    'type': 'string',
                    'title': 'Path to CA certificate file to verify peer against (SSL)'
                },
                'ssl_cert': {
                    'type': 'string',
                    'title': 'Path to client certificate file (SSL)'
                },
                'ssl_key': {
                    'type': 'string',
                    'title': 'Path to private key file (SSL)'
                }
            })

        return schema

    def run_query(self, query, user):
        import MySQLdb

        params = self.configuration['params'].split(',')

        all_columns = []
        all_rows = []
        error = None

        for param in params:
            param = param.strip()
            connection = None
            try:
                config = {
                    'host': self.configuration.get('host', ''),
                    'db': self.configuration['db'],
                    'port': self.configuration.get('port', '3306'),
                    'user': self.configuration.get('user', ''),
                    'passwd': self.configuration.get('passwd', '')
                }

                for key in config:
                    if key == 'port':
                        config[key] = int(config[key].replace('{param}', param))
                    else:
                        config[key] = config[key].replace('{param}', param)

                connection = MySQLdb.connect(host=config['host'],
                                             user=config['user'],
                                             passwd=config['passwd'],
                                             db=config['db'],
                                             port=config['port'],
                                             charset='utf8', use_unicode=True,
                                             ssl=self._get_ssl_parameters(),
                                             connect_timeout=60)
                cursor = connection.cursor()
                logger.debug("MySQL running query: %s", query)
                cursor.execute(query)

                sharding_data = cursor.fetchall()

                while cursor.nextset():
                    sharding_data = cursor.fetchall()

                if cursor.description is not None:
                    columns = self.fetch_columns([(i[0], types_map.get(i[1], None)) for i in cursor.description])
                    rows = [dict(zip((c['name'] for c in columns), row)) for row in sharding_data]

                    sharding_data = {'columns': columns, 'rows': rows}

                    all_columns = sharding_data['columns']
                    if self.configuration.get('show_params'):
                        all_columns.insert(0, {"type": "string", "friendly_name": "database", "name": "database"})

                    for row in sharding_data['rows']:
                        if self.configuration.get('show_params'):
                            row['database'] = param
                        all_rows.append(row)

                cursor.close()
            except MySQLdb.Error as e:
                if error is None:
                    error = ""
                error += param + ": " + e.args[1] + " "
            except KeyboardInterrupt:
                cursor.close()
                if error is None:
                    error = ""
                error += param + ": " + "Query cancelled by user. "
            finally:
                if connection:
                    connection.close()

        if len(all_rows) is 0:
            if error is None:
                error = ""
            error += "No data was returned."

        data = {'columns': all_columns, 'rows': all_rows}
        json_data = json_dumps(data)

        return json_data, error


class ShardingMysqlAggregate(ShardingMysql):
    @classmethod
    def name(cls):
        return "MySQL (Sharding, Aggregate)"

    @classmethod
    def type(cls):
        return 'sharding_mysql_aggregate'

    @classmethod
    def enabled(cls):
        return True

    @classmethod
    def annotate_query(cls):
        return True

    @classmethod
    def configuration_schema(cls):
        show_ssl_settings = parse_boolean(os.environ.get('MYSQL_SHOW_SSL_SETTINGS', 'true'))

        schema = {
            'type': 'object',
            'properties': {
                'params': {
                    'type': 'string',
                    'default': 'shard1, shard2, shard3',
                    'title': 'Shard Parameter (Replace on {param})'
                },
                'aggregate_columns': {
                    'type': 'number',
                    'default': 1,
                    'title': 'Columns used as aggregation keys (N columns from the left)'
                },
                'host': {
                    'type': 'string',
                    'default': '127.0.0.1'
                },
                'user': {
                    'type': 'string'
                },
                'passwd': {
                    'type': 'string',
                    'title': 'Password'
                },
                'db': {
                    'type': 'string',
                    'title': 'Database name',
                    'default': 'test_{param}'
                },
                'port': {
                    'type': 'string',
                    'default': '3306',
                }
            },
            "order": ['params', 'aggregate_columns', 'host', 'port', 'user', 'passwd', 'db'],
            'required': ['db'],
            'secret': ['passwd']
        }

        if show_ssl_settings:
            schema['properties'].update({
                'use_ssl': {
                    'type': 'boolean',
                    'title': 'Use SSL'
                },
                'ssl_cacert': {
                    'type': 'string',
                    'title': 'Path to CA certificate file to verify peer against (SSL)'
                },
                'ssl_cert': {
                    'type': 'string',
                    'title': 'Path to client certificate file (SSL)'
                },
                'ssl_key': {
                    'type': 'string',
                    'title': 'Path to private key file (SSL)'
                }
            })

        return schema

    def run_query(self, query, user):
        json_data, error = super(ShardingMysqlAggregate, self).run_query(query, user)
        data = json_loads(json_data)
        if error is None:
            aggregate_columns = self.configuration.get('aggregate_columns', 1)

            ordered_columns = []
            for ordered_column in data['columns']:
                ordered_columns.append(ordered_column['name'])

            aggregated_dict = {}
            for row in data['rows']:
                ordered = []
                for ordered_column in ordered_columns:
                    ordered.append(row[ordered_column])
                keys = []
                values = []
                i = 0
                for value in ordered:
                    i += 1
                    if i <= aggregate_columns:
                        keys.append(value)
                    else:
                        if type(value) is str:
                            raise Exception('String found in aggregate column.')
                        values.append(value)

                if len(keys) < aggregate_columns or len(values) is 0:
                    raise Exception('Aggregate query is incorrect.')

                self.aggregate(aggregated_dict, keys, values)

            flattened_dicts = []
            self.flatten(aggregated_dict, ordered_columns, {}, flattened_dicts)
            logger.info(flattened_dicts)
            data['rows'] = flattened_dicts
        return json_dumps(data), error

    def aggregate(self, target_dict, keys, values):
        key = keys.pop(0)
        if len(keys) is not 0:
            if key not in target_dict:
                target_dict[key] = {}

            self.aggregate(target_dict[key], keys, values)
        else:
            if key in target_dict:
                a = target_dict[key]
                b = np.array(values)
                target_dict[key] = a + b
            else:
                target_dict[key] = np.array(values)

    def flatten(self, target_dict, ordered_columns, parent_values, result):
        if type(target_dict) is np.ndarray:
            # for
            values = {}
            for k, v in parent_values.items():
                values[k] = v

            i = 0
            for column in ordered_columns:
                values[column] = target_dict[i]
                i += 1

            result.append(values)
        else:
            for key, value in target_dict.items():
                parent_values[ordered_columns[0]] = key
                self.flatten(value, ordered_columns[1:], parent_values, result)


register(ShardingMysqlAggregate)
register(ShardingMysql)