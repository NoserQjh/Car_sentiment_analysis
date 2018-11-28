from Fine_grained.Src.Fine_grained.FINE_GRAINED import analysis_comment
import multiprocessing as mp
from time import time, sleep
from global_var import gl
import csv
import codecs


def analysis_process(pid, tasks, task_num, state_queue, unlabel_queue, child_conn, request_queue):
    process_result = []
    unlabeled_text = []
    init_data = pipe_request(request_queue=request_queue, child_conn=child_conn, id=pid, var_name='INIT_DATA')
    # model = pipe_request(request_queue=request_queue, child_conn=child_conn, id=pid, var_name='WORD2VEC_MODEL')
    from Fine_grained.Src.Fine_grained.sentiment_classify import dic_change
    dic_change(init_data['entity_vector_dic'], init_data['relation_dic'], init_data['weight_dic'])
    while True:
        if tasks.empty():
            break
        try:
            text = tasks.get(False)
        except Exception as e:
            continue
        pipe_request(request_queue=request_queue, child_conn=child_conn, id=pid, var_name='PROGRESS',
                     value=round(100 * (task_num - tasks.qsize()) / task_num), type='post')
        _, single_result = analysis_comment(pid=pid, text=text, init_data=init_data, unlabeled_text=unlabeled_text)
        process_result.extend(single_result)
    state_queue.put(process_result)
    unlabel_queue.put(unlabeled_text)
    pipe_request(request_queue=request_queue, child_conn=child_conn, id=pid, var_name=None, type='close')


def pipe_request(request_queue, child_conn, id, var_name, value=None, type='get'):
    child_conn.send([var_name, value, type])
    request_queue.put(id)
    if type == 'get':
        return child_conn.recv()
    return None


def pip_response(request_queue, parent_conns, thread_num):
    finish_pid = []
    while True:
        try:
            id = request_queue.get(False)
            request = parent_conns[id].recv()
            var_name = request[0]
            value = request[1]
            type = request[2]
            if type == 'get':
                value = gl.get_value(var_name)
                parent_conns[id].send(value)
            elif type == 'post':
                gl.set_value(var_name, value)
                # print('STATE:%s PROGRESS:%d"' % (gl.get_value('STATE'), gl.get_value('PROGRESS')))
            elif type == 'close':
                finish_pid.append(id)
                if len(finish_pid) == thread_num:
                    break
        except Exception as e:
            sleep(1)
            continue


def load_exist_state_list(file_path):
    from pickle import load
    f = open(file_path, 'rb')
    state_list = load(f)
    review_num = load(f)
    words_num = load(f)
    return state_list, review_num, words_num


def save_satate_list(review_num, words_num, state_list, file_path):
    from pickle import dump
    f = open(file_path, 'wb')
    dump(state_list, f)
    dump(review_num, f)
    dump(words_num, f)
    f.close()


def is_state_file(file_name):
    import re
    if re.search('statelist_*\d*.txt$', file_name) is None:
        return False
    else:
        return True


def gen_summary(texts=None, filename=None, thread_num=mp.cpu_count()):
    if not texts:
        texts = load_texts(filename)
    # # single process
    # text='\n'.join(texts)
    # start=time()
    # _, state_list = analysis_comment( text=text, init_data=gl.get_value('INIT_DATA'), model=gl.get_value('WORD2VEC_MODEL'))
    # print('multi-analysis by %d process, %d comments time use: %ds' % (thread_num, len(texts), time() - start))
    upload_file_name = gl.get_value('UPLOAD_FILE_PATH')
    if is_state_file(upload_file_name):
        state_list, review_num, words_num = load_exist_state_list(u'./uploads/' + upload_file_name)
    else:
        words_num = 0
        review_num = 0
        try:
            with open('./uploads/' + gl.get_value('UPLOAD_FILE_PATH'), 'r', encoding='utf8') as fr:
                for line in fr:
                    words_num += len(line.strip())
                    review_num += 1
        except:
            pass
        if review_num == 0:
            review_num = 1

        ctx = mp.get_context('spawn')
        tasks = ctx.Queue(len(texts))
        state_queue = ctx.Queue(thread_num)
        unlabel_queue = ctx.Queue(thread_num)
        parent_conns = []
        request_queue = ctx.Queue()
        thread_list = []
        for text in texts:
            tasks.put(text)
        task_num = tasks.qsize()
        # gl.set_value('INIT_DATA',init_data)
        start = time()
        for i in range(thread_num):
            parent_conn, child_conn = ctx.Pipe()
            parent_conns.append(parent_conn)
            p = ctx.Process(target=analysis_process,
                            args=(i, tasks, task_num, state_queue, unlabel_queue, child_conn, request_queue,))
            thread_list.append(p)
        for p in thread_list:
            p.start()
        pip_response(request_queue, parent_conns, thread_num)
        # for p in thread_list:
        #     p.join()
        gl.set_value('PROGRESS', 100)
        print('single analysis by %d process, %d comments time use: %ds' % (thread_num, task_num, time() - start))
        # p=Process(target=analysis_process,args=(tasks,tasks.qsize(),state_list,lock,init_data))
        # p.start()
        # p.join()

        state_list = []
        while not state_queue.empty():
            state_list.extend(state_queue.get())
        save_satate_list(review_num, words_num, state_list, u'static/result_state_list/' + gl.get_value(
            'PRODUCT') + '/' + upload_file_name + '.statelist.txt')
        unlabeled_text = gl.get_value('UNLABELED_TEXT')
        while not unlabel_queue.empty():
            unlabeled_text.extend(unlabel_queue.get())
    ent_attr_polar, ent_attr_text, attr_description, word_freq = summary_dicts(state_list)
    gl.set_value('REVIEW_NUM', review_num)
    gl.set_value('WORDS_NUM', words_num)
    gl.set_value('ENT_ATTR_POLAR', ent_attr_polar)
    gl.set_value('ENT_ATTR_TEXT', ent_attr_text)
    gl.set_value('ATTR_DESCRIPTION', attr_description)
    gl.set_value('WORD_FREQ', word_freq)

    headers = ['实体', '属性', '评价极性', '评价数目', '评价占比', '评价示例']
    rows = []
    for ent_attr, polars in ent_attr_polar.items():
        ent = ent_attr.split('-')[0]
        attr = ent_attr.split('-')[1]
        total = polars[0] + polars[1] + polars[2]
        if total == 0:
            total = 1
        rows.append({'实体': ent, '属性': attr, '评价极性': '正面', '评价数目': polars[0],
                     '评价占比': polars[0] / total,
                     '评价示例': ' || '.join(ent_attr_text[ent_attr][0])})
        rows.append({'实体': ent, '属性': attr, '评价极性': '中性', '评价数目': polars[1],
                     '评价占比': polars[1] / total,
                     '评价示例': ' || '.join(ent_attr_text[ent_attr][1])})

        rows.append({'实体': ent, '属性': attr, '评价极性': '负面', '评价数目': polars[2],
                     '评价占比': polars[2] / total,
                     '评价示例': ' || '.join(ent_attr_text[ent_attr][2])})
    csv_filepath = filename.replace('.txt', '.csv')
    with codecs.open('./static/downloads/' + csv_filepath, 'w', 'utf-8-sig') as fw:
        f_csv = csv.DictWriter(fw, headers)
        f_csv.writeheader()
        f_csv.writerows(rows)
    return csv_filepath


def summary_dicts(state_list):
    ent_attr_polar = dict()
    ent_attr_text = dict()
    attr_description = dict()
    word_freq = dict()
    for state in state_list:
        ent = state.this_entity_name
        attr = state.this_attribute_name
        polar = state.this_score
        txt = state.text
        attr_polars = ent_attr_polar.setdefault(ent + '-' + attr, [0, 0,
                                                                   0])  # value is the number of positive/neural/negative reviews of the attribute
        txts = ent_attr_text.setdefault(ent + '-' + attr,
                                        [[], [], []])  # value is the set of pos/neu/neg reviews of the entity
        description = state.this_va
        attr_descriptions = attr_description.setdefault(attr, [[], [], []])
        if polar == 1:
            attr_descriptions[0].append(description)
            for word in [ent, attr, description]:
                freq = word_freq.setdefault(word, [0, 0, 0])
                freq[0] = freq[0] + 1
            if txt not in txts[0]:
                attr_polars[0] = attr_polars[0] + 1
                txts[0].append(txt)
        elif polar == 0:
            attr_descriptions[1].append(description)
            for word in [ent, attr, description]:
                freq = word_freq.setdefault(word, [0, 0, 0])
                freq[1] = freq[1] + 1
            if txt not in txts[1]:
                attr_polars[1] = attr_polars[1] + 1
                txts[1].append(txt)
        elif polar == -1:
            attr_descriptions[2].append(description)
            for word in [ent, attr, description]:
                freq = word_freq.setdefault(word, [0, 0, 0])
                freq[2] = freq[2] + 1
            if txt not in txts[2]:
                attr_polars[2] = attr_polars[2] + 1
                txts[2].append(txt)
        else:
            pass
        if attr_polars == [0, 0, 0]:
            ent_attr_polar.pop(ent + '-' + attr)
            ent_attr_text.pop(ent + '-' + attr)
            # ent_attr_polar[ent+'-'+attr]=attr_polars
            # ent_attr_text[ent+'-'+attr]=txts
    return ent_attr_polar, ent_attr_text, attr_description, word_freq


def load_texts(filename):
    texts = []
    try:
        with open('./uploads/' + filename, 'r', encoding='utf8') as fr:
            for line in fr:
                texts.append(line.strip())
    except:
        pass
    return texts


def main():
    pass


if __name__ == '__main__':
    main()

    print('\nProcess finished')
