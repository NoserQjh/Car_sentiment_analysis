# -*- coding: UTF-8 -*-
from flask import Flask, request, render_template, make_response

from global_var import gl
from knowledge_base import knowledge_base_init

app = Flask(__name__)


def gloabal_var_init(product='汽车'):
    gl.set_value('PRODUCT', product)
    gl.set_value('KNOWLEDGE_BASE', knowledge_base_init(product))


@app.route('/index', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/knowledge_graph_test', methods=['GET', 'POST'])
def knowledge_graph():
    product = gl.get_value("PRODUCT", '汽车')
    return render_template("knowledge_graph.html", product=product)


@app.route('/knowledge_base', methods=['GET', 'POST'])
def kb_graph():
    entity = request.args.get('entity')
    attribute = request.args.get('attribute')
    knowledge_base = gl.get_value('KNOWLEDGE_BASE')
    if entity == '0':
        entity = None
    if attribute == '0':
        attribute = None
    if entity is None:
        entity = knowledge_base.productName
    knowledge_base.write_whole_part_info(entity)
    resp = make_response(
        render_template('knowledge_base.html', ent=entity, attr=attribute, product=gl.get_value('PRODUCT', '汽车')))
    resp.cache_control.max_age = -1

    return resp


if __name__ == '__main__':
    print('Server is running')
    gloabal_var_init()
    app.run(host='0.0.0.0', debug=False, port=5001, threaded=True)  # debug=True
