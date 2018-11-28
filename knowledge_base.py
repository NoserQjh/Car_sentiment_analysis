# -*- coding: UTF-8 -*-
import os
import json
import pickle


class KnowledgeBaseConfig:
    def __init__(self, product='汽车'):
        self.LIB_PATH = os.path.join(os.path.abspath('./KnowledgeBase'), product)
        self.ENTITY_WORD_PATH = os.path.join(self.LIB_PATH, 'WordEntity.txt')
        self.ATTRIBUTE_WORD_PATH = os.path.join(self.LIB_PATH, 'WordAttribute.txt')
        self.DESCRIPTION_WORD_PATH = os.path.join(self.LIB_PATH, 'WordDescription.txt')
        self.WHOLE_PART_PATH = os.path.join(self.LIB_PATH, 'WholePart.txt')
        self.ENTITY_ATTRIBUTE_PATH = os.path.join(self.LIB_PATH, 'EntityAttribute.txt')
        self.TARGET_DESCRIPTION_SENTIMENT_PATH = os.path.join(self.LIB_PATH, 'TargetDescriptionSentiment.txt')
        self.ENTITY_SYNONYM_PATH = os.path.join(self.LIB_PATH, 'EntitySynonym.txt')
        self.ATTRIBUTE_SYNONYM_PATH = os.path.join(self.LIB_PATH, 'AttributeSynonym.txt')
        self.JS_CACHE_PATH = os.path.abspath('./static/kb_json')

        self.CACHE_PATH = os.path.join(self.LIB_PATH, 'KnowledgeBaseCache.pkl')
        self.LOAD_FROM_CACHE = False  # load knowledge base from cached pikle file

        self.PROB_DA_PATH = os.path.join(self.LIB_PATH, 'ProbDA.txt')


class KnowledgeBase:
    def __init__(self):
        self.productName = ""  # Product Name, also the top level of entity
        self.entitySet = set()
        self.attributeSet = set()
        self.descriptionSet = set()
        self.pairES = dict()  # Entity-Synonym pair, Key: Entity, Value: Set of Synonyms for this Entity
        self.pairSE = dict()  # Synonym-Entity pair, Key: Synonym, Value: Entity for this Synonym
        self.pairAS = dict()  # Attribute-Synonym pair, Key: Attribute, Value: Set of Synonyms for this Attribute
        self.pairSA = dict()  # Synonym-Attribute pair, Key: Synonym, Value: Attribute for this Synonym
        self.pairEA = dict()  # Entity-Attribute pair, Key: Entity, Value: Set of Attributes for this Entity
        self.pairAE = dict()  # Attribute-Entity pair, Key: Attribute, Value: Set of Entities for this Attribute
        self.pairWP = dict()  # Whole-Part entity pair, Key: Entity, Value: Set of Child-Entities for this Entity
        self.pairPW = dict()  # Part-Whole entity pair, Key: Entity, Value: Set of Father-Entities for this Entity
        self.pairTD = dict()  # Target-Description pair, Key: Target(Entity or Attribute), Value: Set of Descriptions for this target
        self.pairDT = dict()  # Description-Target pair, Key: Description, Value: Set of Target for this Description
        self.pairSentiment = dict()  # Target,Description-Sentiment pair, Key: (Target,Description), Value: Sentiment (POS/NEU/NEG)
        self.js_cache_path = ""
        self.probDA = dict()  # Description&Attribute to probability Key: [Description, Attribute] Value: Probability

    def save(self, cache_path):
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_cache(cache_path):
        with open(cache_path, 'rb') as f:
            kb = pickle.load(f)
        return kb

    def load_knowledge_base(self, entity_word_dir, attribute_word_dir, description_word_dir, whole_part_dir,
                            entity_attribute_dir, target_description_sentiment_dir, entity_synonym_dir,
                            attribute_synonym_dir, js_cache_path, prob_da_dir, product_name="汽车"):
        """
        load Knowledge Base from files
        :param entity_word_dir: 
        :param attribute_word_dir: 
        :param description_word_dir: 
        :param whole_part_dir: 
        :param entity_attribute_dir: 
        :param target_description_sentiment_dir: 
        :param entity_synonym_dir: 
        :param attribute_synonym_dir: 
        :param product_name: 
        :return: 
        """
        self.productName = product_name
        self.js_cache_path = os.path.join(js_cache_path, self.productName)
        if os.path.exists(self.js_cache_path) is False:
            os.makedirs(self.js_cache_path)

        def load_word_file(word_file_dir):
            word_set = set()
            with open(word_file_dir, encoding='utf8') as f:
                all_lines = f.readlines()
            for line in all_lines:
                s = line.strip()
                if s == '':
                    continue
                word_set.add(s)
            return word_set

        self.entitySet = load_word_file(entity_word_dir)
        self.attributeSet = load_word_file(attribute_word_dir)
        self.descriptionSet = load_word_file(description_word_dir)

        def load_pair(pair_file_dir, key_set, value_set=None):
            key_value_dict = dict()
            for key in key_set:
                key_value_dict.setdefault(key, set())
            with open(pair_file_dir, encoding='utf8') as f:
                all_lines = f.readlines()
            for line in all_lines:
                pair = line.strip().split()
                if len(pair) < 2:
                    continue
                key = pair[0]
                value = pair[1]
                if key not in key_set or (value_set is not None and value not in value_set):
                    continue
                key_value_dict[key].add(value)

            value_key_dict = dict()
            if value_set is None:
                value_set = set()
                for values in key_value_dict.values():
                    value_set = value_set | values
            for value in value_set:
                value_key_dict.setdefault(value, set())
            for key, values in key_value_dict.items():
                for value in values:
                    value_key_dict[value].add(key)
            return key_value_dict, value_key_dict

        self.pairES, self.pairSE = load_pair(entity_synonym_dir, self.entitySet)
        self.pairAS, self.pairSA = load_pair(attribute_synonym_dir, self.attributeSet)
        self.pairWP, self.pairPW = load_pair(whole_part_dir, self.entitySet, self.entitySet)
        self.pairEA, self.pairAE = load_pair(entity_attribute_dir, self.entitySet, self.attributeSet)
        self.pairTD, self.pairDT = load_pair(target_description_sentiment_dir, self.entitySet | self.attributeSet,
                                             value_set=self.descriptionSet)

        def load_sentiment():
            with open(target_description_sentiment_dir, 'r', encoding='utf8') as f:
                all_lines = f.readlines()
            for line in all_lines:
                words = line.strip().split()
                if len(words) < 3:
                    continue
                target, description, sentiment = tuple(words)
                if sentiment not in ['POS', 'NEU', 'NEG']:
                    continue
                self.pairSentiment[(target, description)] = sentiment

        load_sentiment()

        # check one synonym with different target; check loop path in entity tree;
        def check_synonyms(synonym_origin_dict, origin_synonym_dict, origin_set):
            for synonym, origins in synonym_origin_dict.items():
                if len(origins) == 0:
                    print("Error! No entity/attribute has synonym word %s" % synonym)
                else:
                    if len(origins) > 1:
                        print("Error! More than one entities/attributes have the same synonym word %s: " % synonym,
                              origins, " only the first one will be remained")
                    while len(origins) > 1:
                        origin = origins.pop()
                        origin_synonym_dict[origin].remove(synonym)
                    origin = origins.pop()
                    synonym_origin_dict[synonym] = origin
            # check if origin words of origin_synonym_dict are in origin set. if not, check if one of the synonym is in
            for origin, synonyms in origin_synonym_dict.items():
                flag = False
                if origin in origin_set:
                    continue
                for synonym in synonyms:
                    if synonym in orgin_set:
                        print("%s is not in origin set, but its synonym %s is in" % (origin, synonym))
                        synonyms.remove(synonym)
                        synonyms.add(origin)
                        origin_synonym_dict.pop(origin)
                        origin_synonym_dict[synonym] = synonyms
                        synonym_origin_dict.pop(synonym)
                        synonym_origin_dict[origin] = synonym
                        flag = True
                        break
                if flag:
                    continue
                print("%s and its synonyms art not in the origin set, will be removed" % origin)
                origin_synonym_dict.pop(origin)
                for synonym in synonyms:
                    synonym_origin_dict.pop(synonym)

        check_synonyms(self.pairSE, self.pairES, self.entitySet)
        check_synonyms(self.pairSA, self.pairAS, self.attributeSet)

        def merge_entity():
            pop_items = []
            for origin, synonyms in self.pairES.items():
                for synonym in synonyms:
                    if synonym not in self.entitySet:
                        continue
                    print("entity %s is synonym of another entity %s, they will be merged" % (origin, synonym))
                    self.entitySet.remove(synonym)
                    pop_items.append(synonym)
                    # entity-attributes pairs merge
                    attributes = self.pairEA[synonym]
                    self.pairEA[origin].update(attributes)
                    for attribute in attributes:
                        self.pairAE[attribute].remove(synonym)
                        self.pairAE[attribute].add(origin)
                    self.pairEA.pop(synonym)
                    # target-description pairs merge
                    descriptions = self.pairTD[synonym]
                    self.pairTD[origin].update(descriptions)
                    for description in descriptions:
                        self.pairDT[description].remove(synonym)
                        self.pairDT[description].add(origin)
                        sentiment = self.pairSentiment[(synonym, description)]
                        self.pairSentiment[(origin, description)] = sentiment
                        self.pairSentiment.pop((synonym, description))
                    self.pairTD.pop(synonym)
                    # Whole-part pairs merge
                    children = self.pairWP[synonym]
                    for child in children:
                        if child != origin:
                            self.pairWP[origin].add(child)
                            self.pairPW[child].remove(synonym)
                            self.pairPW[child].add(origin)
                        else:
                            self.pairPW[origin].remove(synonym)
                    self.pairWP.pop(synonym)
                    fathers = self.pairPW[synonym]
                    for father in fathers:
                        if father != origin:
                            self.pairPW[origin].add(father)
                            self.pairWP[father].remove(synonym)
                            self.pairWP[father].add(origin)
                        else:
                            self.pairWP[origin].remove(synonym)
                    self.pairPW.pop(synonym)
            for item in pop_items:
                self.pairES.pop(item)

        def merge_attribute():
            pop_items = []
            for origin, synonyms in self.pairAS.items():
                for synonym in synonyms:
                    if synonym not in self.attributeSet:
                        continue
                    print("attribute %s is synonym of another attribute %s, they will be merged" % (origin, synonym))
                    self.attributeSet.remove(synonym)
                    pop_items.append(synonym)
                    # entity-attribute pairs merge
                    entities = self.pairAE[synonym]
                    self.pairAE[origin].update(entities)
                    for entity in entities:
                        self.pairEA[entity].remove(synonym)
                        self.pairEA[entity].add(origin)
                    self.pairAE.pop(synonym)
                    # target-description pairs merge
                    descriptions = self.pairTD[synonym]
                    self.pairTD[origin].update(descriptions)
                    for description in descriptions:
                        self.pairDT[description].remove(synonym)
                        self.pairDT[description].add(origin)
                        sentiment = self.pairSentiment[(synonym, description)]
                        self.pairSentiment[(origin, description)] = sentiment
                        self.pairSentiment.pop((synonym, description))
                    self.pairTD.pop(synonym)
            for item in pop_items:
                self.pairAS.pop(item)

        merge_entity()
        merge_attribute()

        def check_whole_part_loop():
            while True:
                path = []
                loop_path, loop_flag = self.check_loop_by_dfs(self.productName, path)
                del path
                if loop_flag is False:
                    break
                print("Error! There is a loop path in whole-part pairs: ", loop_path,
                      " pair (%s, %s) will be removed" % (loop_path[-2], loop_path[-1]))
                self.remove_whole_part_pair(loop_path[-2], loop_path[-1], True)

        # check_whole_part_loop()

        def load_ProbDA(prob_da_dir, probDA_dict):
            try:
                with open(prob_da_dir, 'r', encoding='UTF-8') as infile:
                    for line in infile:
                        if line[0] == ' ':
                            continue
                        line = line.strip()
                        line = line.split('\t')
                        probDA_dict[(line[2], line[0])] = int(line[3])
            except Exception as e:
                print(e)

        load_ProbDA(prob_da_dir, self.probDA)
        self.write_js_variables(True)

    def check_loop_by_dfs(self, entity, path=[]):
        """
        check loop path in entity tree by DFS
        :param entity: (string) input entity node to check loop path
        :param path: (list) input path from father entity
        :return: (list, bool) the first loop path and whether there is a loop path
        """
        if entity in path:
            loop_flag = True
            loop_path = path[path.index(entity):] + [entity]
            return loop_path, loop_flag
        path.append(entity)
        children = self.pairWP[entity]
        for child_entity in children:
            child_loop_path, child_loop_flag = self.check_loop_by_dfs(child_entity, path)
            if child_loop_flag is True:
                return child_loop_path, child_loop_flag
        path.pop()
        return [], False

    # ----functions to write js files to save variables for clients---- #
    def write_js_file(self, json_data, var_name, file_name, over_write=False):
        file_path = os.path.join(self.js_cache_path, file_name + '.js')
        if over_write is False and os.path.exists(file_path) is True:
            return
        if len(json_data) != len(var_name):
            print(
                'Error when try to write file %s, the number of variables not equals to the json data list' % file_path)
        with open(file_path, 'wb') as f:
            for name, data in zip(var_name, json_data):
                f.write((name + '=\'' + json.dumps(data) + '\';').replace('\\"', '\\\\"').encode('utf-8'))

    def write_js_variables(self, over_write=False):
        # ---- Whole-Part relationship between Entities---- #
        self.write_whole_part_info(self.productName, 2, 0)
        # ---- Entity information (Attributes, Synonyms, and (Description, Sentiment) pairs) ---- #
        for _entity in self.entitySet:
            self.write_entity_info(_entity, over_write)
        # ---- Attribute information (Synonyms and (Description, Sentiment) pairs) ---- #
        for _attribute in self.attributeSet:
            self.write_attribute_info(_attribute, over_write)
        for _description in self.descriptionSet:
            self.write_description_info(_description, over_write)

    def write_whole_part_info(self, entity, child_level=1, father_level=0):
        entity_tree, _ = self.build_whole_part_tree(entity, 1, child_level=child_level, father_level=father_level)
        if entity_tree is not None:
            self.write_js_file([entity_tree], ['whole_part'], 'whole_part_%s' % entity, True)

    @staticmethod
    def build_2level_tree(father, children, father_type, children_type):
        id_num = 1
        father_node = {'name': father, 'child': [], 'id': id_num, 'type': father_type}
        for child in children:
            id_num = id_num + 1
            child_node = {'name': child, 'child': [], 'id': id_num, 'type': children_type}
            father_node['child'].append(child_node)
        return father_node

    def build_whole_part_tree(self, entity, id_num=1, child_level=1, father_level=1):
        if not self.have_entity(entity, True):
            print("failed to write whole-part variable: %s is not an entity" % entity)
            return None, id_num
        root_node = {'name': entity, 'child': [], 'father': [], 'id': id_num, 'type': 'entity'}
        id_num = id_num + 1
        if child_level == 0 and father_level == 0:
            return root_node, id_num
        if child_level > 0:
            children = self.children_of_entity(entity, False)
            for child in children:
                child_node, id_num = self.build_whole_part_tree(child, id_num, child_level - 1, 0)
                root_node['child'].append(child_node)
        if father_level > 0:
            fathers = self.father_of_entity(entity, False)
            for father in fathers:
                father_node, id_num = self.build_whole_part_tree(father, id_num, 0, father_level - 1)
                root_node['father'].append(father_node)
        return root_node, id_num

    # todo: rename opinion to description, polar to sentiment, describe to title

    @staticmethod
    def build_target_sentiment_tree(target, description_sentiment_pairs, target_type):
        target_node = {'name': target, 'child': [], 'id': 1, 'type': target_type}
        pos_node = {'name': '正向描述', 'child': [], 'id': 2, 'type': 'describe'}
        neu_node = {'name': '中性描述', 'child': [], 'id': 3, 'type': 'describe'}
        neg_node = {'name': '负向描述', 'child': [], 'id': 4, 'type': 'describe'}
        id = 4
        for description, sentiment in description_sentiment_pairs:
            id = id + 1
            if sentiment == "POS":
                pos_node['child'].append(
                    {'name': description, 'child': [], 'id': id, 'type': 'opinion', 'polar': 1})
            if sentiment == "NEG":
                neg_node['child'].append(
                    {'name': description, 'child': [], 'id': id, 'type': 'opinion', 'polar': -1})
            if sentiment == "NEU":
                neu_node['child'].append(
                    {'name': description, 'child': [], 'id': id, 'type': 'opinion', 'polar': 0})
        target_node['child'].append(pos_node)
        target_node['child'].append(neu_node)
        target_node['child'].append(neg_node)
        return target_node

    def write_entity_info(self, entity, over_write=True):
        """
        write entity's information (Attributes, Synonyms, and (Description, Sentiment) pairs) as javascript variables in a js file named by it
        :param entity: (string) input entity
        :param over_write: (bool) whether to overwrite exists file
        :return:
        """
        if self.have_entity(entity) is False:
            # check failure
            print("failed to write info about %s, it's not an entity" % entity)

        if over_write is False and os.path.exists(os.path.join(self.js_cache_path, entity + '.js')):
            return

        attributes = self.attributes_of_entity(entity, False)
        ent_attr = self.build_2level_tree(entity, attributes, 'entity', 'attribute')

        synonyms = self.synonyms_of_entity(entity)
        ent_synonym = self.build_2level_tree(entity, synonyms, 'entity', 'entity_synonym')

        descriptions = list(self.descriptions_of_target(entity, True))
        sentiments = [self.sentiment_of_target_description_pair(entity, x) for x in descriptions]
        ent_sentiment = self.build_target_sentiment_tree(entity, zip(descriptions, sentiments), 'entity')

        synonym_group = {'name': "synonyms",
                         'children': [{'name': x, 'type': "entity_synonym"} for x in self.pairES[entity]],
                         'type': "group"}
        father_group = {'name': "fathers", 'children': [{'name': x, 'type': "entity"} for x in self.pairPW[entity]],
                        'type': "group"}
        children_group = {'name': "children", 'children': [{'name': x, 'type': "entity"} for x in self.pairWP[entity]],
                          'type': "group"}
        attribute_group = {'name': "attributes",
                           'children': [{'name': x, 'type': "attribute"} for x in self.pairEA[entity]],
                           'type': "group"}
        pos_description_group = {'name': "positive",
                                 'children': [{'name': x, 'type': "positive_description", 'sentiment': "POS"}
                                              for x in self.pairTD[entity]
                                              if self.sentiment_of_target_description_pair(entity, x) == "POS"],
                                 'type': "group"}
        neu_description_group = {'name': "neutral",
                                 'children': [{'name': x, 'type': "neutral_description", 'sentiment': "NEU"}
                                              for x in self.pairTD[entity]
                                              if self.sentiment_of_target_description_pair(entity, x) == "NEU"],
                                 'type': "group"}
        neg_description_group = {'name': "negative",
                                 'children': [{'name': x, 'type': "negative_description", 'sentiment': "NEG"}
                                              for x in self.pairTD[entity]
                                              if self.sentiment_of_target_description_pair(entity, x) == "NEG"],
                                 'type': "group"}
        description_group = {'name': "descriptions",

                             'children': [pos_description_group, neu_description_group, neg_description_group],
                             'type': "group"}

        entity_graph = [children_group, attribute_group, father_group, description_group, synonym_group]

        self.write_js_file([ent_attr, ent_synonym, ent_sentiment, entity_graph],
                           ['ent_attr', 'ent_synonym', 'ent_sentiment', 'partial_graph'], entity,
                           over_write)

    # todo: add entity_descriptions

    def write_attribute_info(self, attribute, over_write=True):
        """
        write attribute's information (Synonyms, and (Description, Sentiment) pairs) as javascript variables in a js file named by it
        :param attribute: (string) input attribute
        :param over_write: (bool) whether to overwrite exists file
        :return:
        """
        if self.have_attribute(attribute) is False:
            # check failure
            print("failed to write info about %s, it's not an attribute" % attribute)

        if over_write is False and os.path.exists(os.path.join(self.js_cache_path, attribute + '.js')):
            return

        synonyms = self.synonyms_of_attribute(attribute)
        attr_synonym = self.build_2level_tree(attribute, synonyms, 'attribute', 'attribute_synonym')

        descriptions = self.descriptions_of_target(attribute, True)
        sentiments = [self.sentiment_of_target_description_pair(attribute, x) for x in descriptions]
        attr_opinion = self.build_target_sentiment_tree(attribute, zip(descriptions, sentiments), 'attribute')

        synonym_group = {'name': "synonyms",
                         'children': [{'name': x, 'type': "attribute_synonym"} for x in self.pairAS[attribute]],
                         'type': "group"}
        entity_group = {'name': "entities", 'children': [{'name': x, 'type': "entity"} for x in self.pairAE[attribute]],
                        'type': "group"}
        pos_description_group = {'name': "positive",
                                 'children': [{'name': x, 'type': "positive_description", 'sentiment': "POS"}
                                              for x in self.pairTD[attribute]
                                              if self.sentiment_of_target_description_pair(attribute, x) == "POS"],
                                 'type': "group"}
        neu_description_group = {'name': "neutral",
                                 'children': [{'name': x, 'type': "neutral_description", 'sentiment': "NEU"}
                                              for x in self.pairTD[attribute]
                                              if self.sentiment_of_target_description_pair(attribute, x) == "NEU"],
                                 'type': "group"}
        neg_description_group = {'name': "negative",
                                 'children': [{'name': x, 'type': "negative_description", 'sentiment': "NEG"}
                                              for x in self.pairTD[attribute]
                                              if self.sentiment_of_target_description_pair(attribute, x) == "NEG"],
                                 'type': "group"}
        description_group = {'name': "descriptions",
                             'children': [pos_description_group, neu_description_group, neg_description_group],
                             'type': "group"}
        attribute_graph = [entity_group, description_group, synonym_group]

        self.write_js_file([attr_synonym, attr_opinion, attribute_graph],
                           ['attr_synonym', 'attr_opinion', 'partial_graph'], attribute, over_write)

    def write_description_info(self, description, over_write=True):
        if not self.have_description(description):
            print("failed to write info about %s, it's not a description" % description)
        pos_target_group = {'name': "positive", 'children': [], 'type': "group"}
        neu_target_group = {'name': "neutral", 'children': [], 'type': "group"}
        neg_target_group = {'name': "negative", 'children': [], 'type': "group"}
        targets = self.pairDT[description]
        for target in targets:
            target_type = self.target_type(target)
            sentiment = self.sentiment_of_target_description_pair(target, description)
            if sentiment == "POS":
                pos_target_group['children'].append({'name': target, 'type': target_type, 'sentiment': "POS"})
            if sentiment == "NEU":
                neu_target_group['children'].append({'name': target, 'type': target_type, 'sentiment': "NEU"})
            if sentiment == "NEG":
                neg_target_group['children'].append({'name': target, 'type': target_type, 'sentiment': "NEG"})
        description_graph = [pos_target_group, neu_target_group, neg_target_group]
        self.write_js_file([description_graph], ['partial_graph'], description, over_write)

    # APIs for knowledge base query ----start----
    # todo: database query, sql or others

    def have_entity(self, word, include_synonyms_flag=False):
        """
        query if the input word is an entity
        :param word: (string) input word
        :param include_synonyms_flag: (bool)whether to include synonyms entities
        :return: (bool) if the input word is an entity in the knowledge base
        """
        have = word in self.entitySet
        if include_synonyms_flag is True:
            have = have | (word in self.pairSE)
        return have

    def have_attribute(self, word, include_synonyms_flag=False):
        """
        query if the input word is an attribute
        :param word: (string) input word
        :param include_synonyms_flag: (bool)whether to include synonyms attributes
        :return: (bool) if the input word is an attribute in the knowledge base
        """
        have = word in self.attributeSet
        if include_synonyms_flag is True:
            have = have | (word in self.pairSA)
        return have

    def have_description(self, word):
        """
        query if the input word is a description
        :param word:  (string) input word
        :return: (bool) if the word is a description in the knowledge base
        """
        return word in self.descriptionSet

    def have_target(self, word, include_synonyms_flag=False):
        """
        query if the input word is an target (i.e. entity or attribute)
        :param word: (string) input word
        :param include_synonyms_flag: (bool)whether to include synonyms
        :return: (bool) if the input word is an target
        """
        return self.have_entity(word, include_synonyms_flag) or self.have_attribute(word, include_synonyms_flag)

    def have_entity_attribute_pair(self, entity, attribute):
        """
        whether the input pair is an entity-attribute pair in the knowledge base, inputs could be synonyms
        :param entity: (string) input entity (synonym)
        :param attribute: (string) input attribute (synonym)
        :return: (bool) True or False
        """
        if self.have_entity(entity, True):
            origin_attribute = self.attribute_of_synonym(attribute, False)
            if origin_attribute is None:
                return False
            return origin_attribute in self.attributes_of_entity(entity, True)
        else:
            return False

    def have_target_description_pair(self, target, description):
        """
        whether the input pair is a target-description pair in the knowledge base, input target could be a synonym
        :param target: (string) input target (synonym)
        :param description: (string) input description
        :return: (bool) True or False
        """
        if self.have_target(target, True):
            return description in self.descriptions_of_target(target, False)
        else:
            return False

    def have_whole_part_pair(self, whole, part):
        """
        whether the input pair is a whole-part pair entity in the knowledge base, input entities could be synonyms
        :param whole: (string) input father entity
        :param part: (string) input child entity
        :return: (bool) True or False
        """
        if self.have_entity(whole, True):
            origin_part = self.entity_of_synonym(part)
            return origin_part in self.children_of_entity(whole, False)
        else:
            return False

    def target_type(self, target):
        """
        query for the target type: entity or attribute or not a target
        :param target: (string) input target word
        :return: (string) target type
        """
        if self.have_entity(target, True):
            return "entity"
        elif self.have_attribute(target, True):
            return "attribute"
        else:
            return None

    def is_synonyms(self, target1, target2):
        """
        whether the input pair are synonyms in the knowledge base
        :param target1: (string) input target1
        :param target2: (string) input target2
        :return: True or False
        """
        origin_target1 = self.target_of_synonym(target1, False)
        if origin_target1 is None:
            return False
        origin_target2 = self.target_of_synonym(target2, False)
        if origin_target2 is None:
            return False
        return origin_target1 == origin_target2

    def synonyms_of_entity(self, entity, check_hint=True):
        """
        return set of synonyms of the input entity, if entity is not in the knowledge base, return None
        :param entity: (string) input entity
        :param check_hint: (bool) whether to print check info
        :return: (set(string)) set of synonyms of the input entity
        """
        if self.have_entity(entity, False):
            return self.pairES[entity]
        else:
            # check failure
            if check_hint:
                print("query synonym failed! %s is not an entity" % entity)
            return None

    def synonyms_of_attribute(self, attribute, check_hint=True):
        """
        return set of synonyms of the input attribute, if attribute is not in the knowledge base, return None
        :param attribute: (string) input attribute
        :param check_hint: (bool) whether to print check info
        :return: (set(string)) set of synonyms of the input attribute
        """
        if self.have_attribute(attribute, False):
            return self.pairAS[attribute]
        else:
            # check failure
            if check_hint:
                print("query synonym failed! %s is not an attribute" % attribute)
            return None

    def synonyms_of_target(self, target, check_hint=True):
        """
        return set of synonyms of the input target, if target is not in the knowledge base, return None
        :param target: (string) input target
        :param check_hint: (bool) whether to print check info
        :return: (set(string)) set of synonyms of the input target
        """
        if self.have_entity(target, False):
            return self.pairES[target]
        elif self.have_attribute(target, False):
            return self.pairAS[target]
        else:
            # check failure
            if check_hint:
                print("query synonym failed! %s is neither an entity nor an attribute" % target)
            return None

    def entity_of_synonym(self, synonym, check_hint=True):
        """
        return the entity of the input synonym word (it could be the entity itself)
        :param synonym: (string) input synonym word
        :param check_hint: (bool) whether to print check failure hints
        :return: (string) the entity of the input synonym word
        """

        if synonym in self.entitySet:
            return synonym
        elif synonym in self.pairSE:
            return self.pairSE[synonym]
        else:
            # check failure
            if check_hint is True:
                print("query entity failed! %s is neither an entity nor an entity synonym" % synonym)
            return None

    def attribute_of_synonym(self, synonym, check_hint=True):
        """
        return the attribute of the input synonym word (it could be the attribute itself)
        :param synonym: (string) input synonym word
        :param check_hint: (bool) whether to print check failure hints
        :return: (string) the attribute of the input synonym word
        """
        if synonym in self.attributeSet:
            return synonym
        elif synonym in self.pairSA:
            return self.pairSA[synonym]
        else:
            if check_hint is True:
                print("query attribute failed! %s is neither an attribute nor an attribute synonym" % synonym)
            return None

    def target_of_synonym(self, synonym, check_hint=True):
        """
        return the target word (entity or attribute) of the input synonym word (it could be the target itself)
        :param synonym: (string) input synonym word
        :param check_hint: (bool) whether to print check failure hints
        :return: (string) the target word of the input synonym word
        """
        entity = self.entity_of_synonym(synonym, False)
        if entity is not None:
            return entity
        attribute = self.attribute_of_synonym(synonym, False)
        if attribute is not None:
            return attribute
        if check_hint is True:
            print("query target failed! %s is neither an target nor an target synonym" % synonym)
        return None

    def children_of_entity(self, entity, include_synonym_flag=False, check_hint=True):
        """
        return set of child-entities of the input entity (synonym), if entity is not in the knowledge base, return None
        :param entity: (string) input entity
        :param include_synonym_flag: (bool) whether to include synonyms in the result set
        :param check_hint: (bool) whether to print check failure hints
        :return: (set(string)) set of child-entities of the input entity
        """
        origin_entity = self.entity_of_synonym(entity, False)
        if origin_entity is None:
            # check failure
            if check_hint:
                print("query child-entity failed! %s is not an entity" % entity)
            return None
        child_entities = self.pairWP[origin_entity]
        if include_synonym_flag is True:
            synonyms = []
            for child in child_entities:
                synonyms.extend(self.synonyms_of_entity(child))
            child_entities.extend(synonyms)
        return child_entities

    def father_of_entity(self, entity, include_synonym_flag=False, check_hint=True):
        """
        return set of father-entities of the input entity (synonym), if entity is not in the knowledge base, return None
        :param entity: (string) input entity (synonym)
        :param include_synonym_flag: (bool) whether to include synonyms in the result set
        :param check_hint: (bool) whether to print check failure hints
        :return: (set(string)) set of father-entities of the input entity
        """
        origin_entity = self.entity_of_synonym(entity, False)
        if origin_entity is None:
            if check_hint:
                print("query father-entity failed! %s is not an entity" % entity)
            return None
        father_entities = self.pairPW[origin_entity]
        if include_synonym_flag is True:
            synonyms = []
            for father in father_entities:
                synonyms.extend(self.synonyms_of_entity(father))
            father_entities.extend(synonyms)
        return father_entities

    def attributes_of_entity(self, entity, include_synonym_flag=False, check_hint=True):
        """
        return set of attributes of the input entity (synonym), if entity is not in the knowledge base, return None
        :param entity: (string) input entity
        :param include_synonym_flag: (bool) whether to include synonyms in the result set
        :param check_hint: (bool) whether to print check info
        :return: (set(string)) set of attributes of the input entity
        """
        origin_entity = self.entity_of_synonym(entity, False)
        if origin_entity is None:
            # check failure
            if check_hint:
                print("query attributes failed! %s is not an entity" % entity)
            return None
        attributes = self.pairEA[origin_entity]
        if include_synonym_flag is True:
            synonyms = []
            for attribute in attributes:
                synonyms.extend(self.synonyms_of_attribute(attribute))
            attributes.extend(synonyms)
        return attributes

    def entities_of_attribute(self, attribute, include_synonym_flag=False, check_hint=True):
        """
        return set of entities of the input attribute (synonym), if attribute is not in the knowledge base, return None
        :param attribute: (string) input attribute word
        :param include_synonym_flag: (bool) whether to include synonyms in the result set
        :param check_hint: (bool) whether to print check info
        :return: (set(string)) set of entities of the input attribute
        """
        origin_attribute = self.attribute_of_synonym(attribute, False)
        if origin_attribute is None:
            # check failure
            if check_hint:
                print("query entities failed! %s is not an attribute" % attribute)
            return None
        entities = self.pairAE[origin_attribute]
        if include_synonym_flag is True:
            synonyms = []
            for entity in entities:
                synonyms.extend(self.synonyms_of_entity(entity))
            entities.extend(synonyms)
        return entities

    def sentiment_of_target_description_pair(self, target, description, check_hint=True):
        """
        return the sentiment of the input (target (synonym), description) pair, i.e. POS/NEU/NEG
        :param target: (string) Entity or Attribute or their synonym
        :param description: (string) input description word
        :param check_hint: (bool) whether to print check info
        :return: (string) Sentiment result, if input is not a pair in knowledge base, return None
        """
        if self.have_description(description) is False:
            # check failure
            if check_hint is True:
                print("query sentiment for pair(%s,%s) failed! %s is not a description" % (
                    target, description, description))
            return None
        origin_target = self.target_of_synonym(target, False)
        if origin_target is None:
            # check failure
            if check_hint is True:
                print("query sentiment for pair(%s,%s) failed! %s is not a target" % (target, description, target))
            return None
        if (origin_target, description) in self.pairSentiment:
            return self.pairSentiment[(origin_target, description)]
        # check failure
        if check_hint is True:
            print("query sentiment for pair(%s,%s) failed! they are not a pair" % (target, description))
        return None

    def descriptions_of_target(self, target, check_hint=True):
        """
        return descriptions of the input target (synonym) word, if target is not in the knowledge base, return None
        :param target: (string) input target
        :param check_hint: (bool) whether to print check info
        :return: (set) descriptions of the input target if with_sentiment_flag
        """
        origin_target = self.target_of_synonym(target, False)
        if origin_target is None:
            if check_hint:
                print("query description failed! %s is not a target" % target)
            return None
        return self.pairTD[origin_target]

    def targets_of_description(self, description, include_synonym_flag=False, check_hint=True):
        """
        return set of (target, sentiment) of the input description word, if description is not in the knowledge base, return None
        :param description: (string) input description word
        :param include_synonym_flag: (bool) whether to include synonyms in the result set
        :param check_hint: (bool) whether to print check info
        :return: (set) targets of the input description
        """
        if description not in self.descriptionSet:
            # check failure
            if check_hint:
                print("query target failed! %s is not a description" % description)
            return None

        targets = self.pairDT[description]
        if include_synonym_flag is True:
            synonyms = []
            for target in targets:
                synonyms.extend(self.synonyms_of_target(target))
            targets.update(synonyms)
        return targets

    # APIs for knowledge base query ----end----

    # APIs for knowledge base modify ----start----
    # todo: database modify, sql or others
    def remove_entity(self, entity, check_hint=True):
        """
        remove the input entity from knowledge base
        :param entity: (string) entity to be removed (it could be a synonym, then remove the origin entity)
        :param check_hint: (bool) whether to print check info
        :return: False for failed, True for succeed
        """
        origin_entity = self.entity_of_synonym(entity)
        if origin_entity is None:
            if check_hint:
                print("failed to remove entity %s: it's not an entity" % entity)
            return False

        # remove whole-part and part-whole pairs
        children = self.pairWP[origin_entity]
        for child in children:
            self.pairPW[child].remove(origin_entity)
        self.pairWP.pop(origin_entity)

        # remove entity-synonym and synonym-entity pairs
        synonyms = self.pairES[origin_entity]
        for synonym in synonyms:
            self.pairSE.pop(synonym)
        self.pairES.pop(origin_entity)

        # remove entity-attribute and attribute-entity pairs
        attributes = self.pairEA[origin_entity]
        for attribute in attributes:
            self.pairAE[attribute].remove(origin_entity)
        self.pairEA.pop(origin_entity)

        # remove target-description and description-target and sentiment pairs
        descriptions = self.pairTD[origin_entity]
        for description in descriptions:
            self.pairDT[description].remove(origin_entity)
            self.pairSentiment.pop((origin_entity, description))
        self.pairTD.pop(origin_entity)

        # remove from entity set
        self.entitySet.remove(origin_entity)
        return True

    def remove_attribute(self, attribute, check_hint=True):
        """
        remove the input attribute from knowledge base
        :param attribute: (string) attribute to be removed (it could be a synonym, then remove the origin attribute)
        :param check_hint: (bool) whether to print check info
        :return: False for failed, True for succeed
        """
        origin_attribute = self.attribute_of_synonym(attribute)
        if origin_attribute is None:
            if check_hint:
                print("failed to remove attribute %s: it's not an attribute" % attribute)
            return False
        # remove attribute-synonym and synonym-attribute pairs
        synonyms = self.pairAS[origin_attribute]
        for synonym in synonyms:
            self.pairSA.pop(synonym)
        self.pairAS.pop(origin_attribute)

        # remove entity-attribute and attribute-entity pairs
        entities = self.pairAE[origin_attribute]
        for entity in entities:
            self.pairEA[entity].remove(origin_attribute)
        self.pairAE.pop(origin_attribute)

        # remove target-description and description-target and sentiment pairs
        descriptions = self.pairTD[origin_attribute]
        for description in descriptions:
            self.pairDT[description].remove(origin_attribute)
            self.pairSentiment.pop((origin_attribute, description))
        self.pairTD.pop(origin_attribute)

        # remove from entity set
        self.attributeSet.remove(origin_attribute)
        return True

    def remove_target(self, target, check_hint=True):
        """
        remove the input target from knowledge base
        :param target: (string) input target to be removed (it could be a synonym, then remove the origin target)
        :param check_hint: (bool) whether to print check info
        :return: False for failed, True for succeed
        """
        target_type = self.target_type(target)
        if target_type is None:
            if check_hint:
                print("failed to remove target %s: it's not a target" % target)
            return False
        if target_type == "entity":
            return self.remove_entity(target)
        else:
            return self.remove_attribute(target)

    def remove_description(self, description, check_hint=True):
        if description not in self.descriptionSet:
            if check_hint:
                print("failed to remove description %s: it's not a description" % description)
            return False
        # remove target-description and description-target and sentiment pairs
        targets = self.pairDT[description]
        for target in targets:
            self.pairTD[target].remove(description)
            self.pairSentiment.pop((target, description))
        self.pairDT.pop(description)
        # remove from description set
        self.descriptionSet.remove(description)
        return True

    def remove_whole_part_pair(self, father, child, check_hint=True):
        """
        remove the (father, child) pair in knowledge base
        :param father: (string) input father entity(synonym)
        :param child: (string) input child entity(synonym)
        :param check_hint: (bool) whether to print check info
        :return: (tuple) the truly removed pair (the input words could be synonyms, return the origin pair), if failed, return None
        """
        if not self.have_whole_part_pair(father, child):
            if check_hint:
                print("remove whole-part pair (%s,%s) failed, this pair not exists" % (father, child))
            return None
        origin_father = self.entity_of_synonym(father, False)
        origin_child = self.entity_of_synonym(child, False)
        self.pairWP[origin_father].remove(origin_child)
        self.pairPW[origin_child].remove(origin_father)
        return origin_father, origin_child

    def remove_entity_attribute_pair(self, entity, attribute, check_hint=True):
        """
        remove the (entity, attribute) pair in knowledge base
        :param entity: (string) input entity(synonym)
        :param attribute: (string) input attribute(synonym)
        :param check_hint: (bool) whether to print check info
        :return: (tuple) the truly removed pair (the input words could be synonyms, return the origin pair), if failed, return None
        """
        if not self.have_entity_attribute_pair(entity, attribute):
            if check_hint:
                print("remove entity-attribute pair (%s,%s) failed, this pair not exists" % (entity, attribute))
            return None
        origin_entity = self.entity_of_synonym(entity)
        origin_attribute = self.attribute_of_synonym(attribute)
        self.pairEA[origin_entity].remove(origin_attribute)
        self.pairAE[origin_attribute].remove(origin_entity)
        return origin_entity, origin_attribute

    def remove_target_synonym_pair(self, target, synonym, check_hint=True):
        """
        remove target-synonym pair in knowledge base
        :param target: (string) input target
        :param synonym: (string) input synonym
        :param check_hint: (bool) whether to print check info
        :return: (tuple) the removed pair, if failed, return None
        """
        if not self.have_target(target, False):
            if check_hint:
                print("failed to remove target-synonym pair (%s, %s): %s is not an origin target" % (
                    target, synonym, target))
            return None
        if synonym == target:
            if check_hint:
                print("failed to remove target-synonym pair (%s, %s): they are the same" % (target, synonym))
            return None
        if not self.is_synonyms(target, synonym):
            if check_hint:
                print("failed to remove target-synonym pair (%s, %s): they are not a pair" % (target, synonym))
            return None
        target_type = self.target_type(target)
        if target_type == "entity":
            self.pairES[target].remove(synonym)
            self.pairSE.pop(synonym)
        else:
            self.pairAS[target].remove(synonym)
            self.pairSA.pop(synonym)
        return target, synonym

    def remove_target_description_pair(self, target, description, check_hint=True):
        """
        remove the (target, description) pair in knowledge base
        :param target: (string) input target(synonym)
        :param description: (string) input description
        :param check_hint: (bool) whether to print check info
        :return: (tuple) the truly removed pair (the input words could be synonyms, return the origin pair), if failed, return None
        """
        if not self.have_target_description_pair(target, description):
            if check_hint:
                print("remove target-description pair (%s,%s) failed, this pair not exists" % (target, description))
            return None
        origin_target = self.target_of_synonym(target)
        self.pairTD[origin_target].remove(description)
        self.pairDT[description].remove(origin_target)
        self.pairSentiment.pop((origin_target, description))
        return origin_target, description

    @staticmethod
    def add_key_value_pair(key, value, key_value_dict, value_key_dict):
        """
        give a pair of (key,value),add value to key_value_dict[key] and add key to value_key_dict[value]
        :param key: (string) input key
        :param value: (string or tuple) input value
        :param key_value_dict: (dict) the key-value dict
        :param value_key_dict: (dict) the value-key dict
        :return:
        """
        if key not in key_value_dict:
            key_value_dict[key] = set()
        key_value_dict[key].add(value)

        if value not in value_key_dict:
            value_key_dict[value] = set()
        value_key_dict[value].add(key)

    def add_new_entity(self, entity, check_hint=True):
        """
        add a single new entity
        :param entity: (string) input new entity
        :param check_hint: (bool) whether to print check info
        :return: (bool) False for failed, True for succeed
        """
        if self.have_target(entity, True):
            # check failure
            if check_hint is True:
                print("failed to add new entity! %s has been in the knowledge base!" % entity)
            return False
        self.entitySet.add(entity)
        self.pairEA[entity] = set()
        self.pairES[entity] = set()
        self.pairWP[entity] = set()
        self.pairPW[entity] = set()
        self.pairTD[entity] = set()
        return True

    def add_new_attribute(self, attribute, check_hint=True):
        """
        add a single new attribute
        :param attribute: (string) input new attribute
        :param check_hint: (bool) whether to print check info
        :return: (bool) False for failed, True for succeed
        """
        if self.have_target(attribute, True):
            if check_hint is True:
                print("failed to add new attribute! %s has been in the knowledge base!" % attribute)
            return False
        self.attributeSet.add(attribute)
        self.pairAE[attribute] = set()
        self.pairAS[attribute] = set()
        self.pairTD[attribute] = set()
        return True

    def add_new_target(self, target, target_type, check_hint=True):
        """
        add a single new target to the knowledge base
        :param target: (string) input target
        :param target_type: (string) "entity" or "attribute"
        :param check_hint: (bool) whether to print check info
        :return: (bool) False for failed, True for succeed
        """
        if self.have_target(target, True):
            if check_hint:
                print("failed to add target %s, already exsits!" % target)
            return False
        if target_type == "entity":
            return self.add_new_entity(target, check_hint)
        elif target_type == "attribute":
            return self.add_new_attribute(target, check_hint)
        else:
            if check_hint:
                print("failed to add target %s, wrong target type: %s (should be 'entity' or 'attribute')" % (
                    target, target_type))
            return False

    def add_new_description(self, description, check_hint=True):
        """
        add a single new attribute
        :param description: (string) input new description
        :param check_hint: (bool) whether to print check info
        :return: (bool) False for failed, True for succeed
        """
        if self.have_description(description):
            if check_hint is True:
                print("failed to add new description! %s has been in the knowledge base!" % description)
            return False
        self.descriptionSet.add(description)
        self.pairDT[description] = set()
        return True

    def add_entity_attribute_pair(self, entity, attribute, able_add=True):
        """
        add entity-attribute pair to knowledge base, if able_add, new entity and attribute can be added
        if the input entity or attribute is a synonym, add the correct pair of origin entity and origin attribute
        :param entity: (string) input entity
        :param attribute: (string) input attribute
        :param able_add: (bool) whether to allow new entity and attribute to be added
        :return: (tuple) the truly added pair (the input words could be synonyms, return the origin pair), if failed, return None
        """
        if not able_add:
            if not self.have_entity(entity) or not self.have_attribute(attribute):
                # check failure
                print(
                    "failed to add entity-attribute pair (%s,%s), one or more of these words are not in the knowledge base" % (
                        entity, attribute))
                return None
        self.add_new_entity(entity, False)
        self.add_new_attribute(attribute, False)
        origin_entity = self.entity_of_synonym(entity)
        origin_attribute = self.attribute_of_synonym(attribute)
        self.add_key_value_pair(origin_entity, origin_attribute, self.pairEA, self.pairAE)
        return origin_entity, origin_attribute

    def add_synonym_pair(self, target1, target2, able_add=True, add_type="entity"):
        """
        add synonym pair to the knowledge base
        if one of the targets has been in the knowledge base, add another one to it
        if both of the targets have been in the knowledge base, check if they are synonym pair
        if none of the targets is existed, add the first target as a new add_type target, then add synonym pair
        :param target1: (string) input target1
        :param target2: (string) input target2
        :param able_add: (bool) whether to add new target when none of the input targets existed
        :param add_type: (string) "entity" or "attribute", type of target to add
        :return:
        """
        origin_target1 = self.target_of_synonym(target1, False)
        origin_target2 = self.target_of_synonym(target2, False)
        if origin_target1 is None and origin_target2 is None:
            # neither of the targets is in the knowledge base
            if not able_add:
                print("failed to add synonym pair (%s, %s), both of the two words are not targets" % (target1, target2))
                return None
            else:
                if add_type == 'entity':
                    return self.add_entity_synonym_pair(target1, target2, True)
                elif add_type == 'attribute':
                    return self.add_attribute_synonym_pair(target1, target2, True)
                else:
                    print(
                        "failed to add synonym pair, wrong add type: %s (should be 'entity' or 'attribute')" % add_type)
                    return None
        elif origin_target2 is None:
            # target1 is in the knowledge base
            target_type = self.target_type(origin_target1)
            if target_type is None:
                print("Unexpected Error! %s is not a target while it should be" % origin_target1)
            if target_type == "entity":
                return self.add_entity_synonym_pair(origin_target1, target2, False)
            else:
                return self.add_attribute_synonym_pair(origin_target1, target2, False)
        elif origin_target1 is None:
            # target2 is in the knowledge base
            target_type = self.target_type(origin_target2)
            if target_type is None:
                print("Unexpected Error! %s is not a target while it should be" % origin_target2)
            if target_type == "entity":
                return self.add_entity_synonym_pair(origin_target2, target1, False)
            else:
                return self.add_attribute_synonym_pair(origin_target2, target1, False)
        else:
            # both is in the knowledge base
            if origin_target1 != origin_target2:
                print("failed to add synonym pair (%s, %s), they are different targets in the knowledge base")
                return None
            else:
                return origin_target1, origin_target1

    def add_entity_synonym_pair(self, entity, synonym, able_add=True):
        """
        add entity-synonym pair to knowledge base, if able_add, new entity can be added
        :param entity: (string) input entity
        :param synonym: (string) input synonym
        :param able_add: (bool) able to add new words to entity set
        :return: (tuple) the truly added pair (input words could be synonyms, return the origin pair), if failed, return None
        """
        origin_entity = self.entity_of_synonym(entity, False)
        if origin_entity is None:
            if not able_add:
                print("failed to add entity-synonym pair (%s,%s): %s is not an entity" % (entity, synonym, entity))
                return None
            if self.add_new_entity(entity) is False:
                print("failed to add entity-synonym pair (%s,%s): %s is an attribute" % (entity, synonym, entity))
                return None
            synonym_set = self.pairES[entity]
            if synonym not in synonym_set:
                synonym_set.add(synonym)
                self.pairSE[synonym] = set([entity])
            return entity, synonym
        else:
            origin_synonym = self.target_of_synonym(synonym, False)
            if origin_synonym is not None:
                if origin_entity != origin_synonym:
                    print("failed to add entity-synonym pair (%s,%s): they are different target in knowledge base" % (
                        entity, synonym))
                    return None
                else:
                    return origin_entity, origin_entity
            else:
                synonym_set = self.pairES[origin_entity]
                if synonym not in synonym_set:
                    synonym_set.add(synonym)
                    self.pairSE[synonym] = set([origin_entity])
                return origin_entity, synonym

    def add_attribute_synonym_pair(self, attribute, synonym, able_add=True):
        """
        add attribute-synonym pair to knowledge base, if able_add, new attribute can be added
        :param attribute: (string) input attribute
        :param synonym: (string) input synonym
        :param able_add: (bool) whether to add new attribute when input attribute is not in the knowledge base
        :return: (tuple) the truly added pair (input words could be synonyms, return the origin pair), if failed, return None
        """
        origin_attribute = self.attribute_of_synonym(attribute, False)
        if origin_attribute is None:
            if not able_add:
                print("failed to add attribute-synonym pair (%s,%s): %s is not an attribute" % (
                    attribute, synonym, attribute))
                return None
            if self.add_new_attribute(attribute) is False:
                print("failed to add attribute-synonym pair (%s,%s): %s is an entity" % (attribute, synonym, attribute))
                return None
            synonym_set = self.pairAS[attribute]
            if synonym not in synonym_set:
                synonym_set.add(synonym)
                self.pairSA[synonym] = set([attribute])
            return attribute, synonym
        else:
            origin_synonym = self.target_of_synonym(synonym, False)
            if origin_synonym is not None:
                if origin_attribute != origin_synonym:
                    print("failed to add entity-synonym pair (%s,%s): they are different target in knowledge base" % (
                        attribute, synonym))
                    return None
                else:
                    return origin_attribute, origin_attribute
            else:
                synonym_set = self.pairAS[origin_attribute]
                if synonym not in synonym_set:
                    synonym_set.add(synonym)
                    self.pairSA[synonym] = set([origin_attribute])
                return origin_attribute, synonym

    def add_whole_part_pair(self, father, child, able_add=True, check_hint=True):
        """
        add a whole-part entity pair to knowledge base
        :param father: (string) input father entity(synonym)
        :param child: (string) input child entity(synonym)
        :param able_add: (bool) whether to add child to entity set when child is not an entity
        :return: (tuple) the truly added pair (input words could be synonyms, return the origin pair), if failed, return None
        """
        origin_father = self.entity_of_synonym(father)
        if origin_father is None:
            if check_hint:
                print("failed to add whole part pair (%s,%s), %s is not an entity %s" % (father, child, father))
            return None
        origin_child = self.entity_of_synonym(child)
        if origin_child is None:
            if not able_add:
                if check_hint:
                    print("failed to add whole part pair (%s,%s), %s is not an entity %s" % (father, child, child))
                return None
            self.add_new_entity(child)
            origin_child = child
        self.pairWP[origin_father].add(origin_child)
        self.pairPW[origin_child].add(origin_father)

        loop_path, loop_flag = self.check_loop_by_dfs(father, [])
        if loop_flag:
            if check_hint:
                print("failed to add-whole part pair (%s, %s), there is a loop path: " % (father, child), loop_path)
            self.remove_whole_part_pair(origin_father, origin_child, True)
            return None
        return origin_father, origin_child

    def add_target_description_sentiment(self, target, description, sentiment, able_overwrite=True,
                                         able_add_target=True, target_type="entity", able_add_description=True,
                                         check_hint=True):
        """
        add target-description-sentiment pair to knowledge base
        :param target: (string) input target (synonym)
        :param description: (string) input description
        :param sentiment: (string) input sentiment (POS/NEU/NEG)
        :param able_overwrite: (bool) whether to overwrite the exist sentiment when (target,description) pair already exists
        :param able_add_target: (bool) whether to add a new target if the input target not exists
        :param target_type: (string) which type of target to add if we need to add the input target to knowledge base
        :param able_add_description: (bool) whether to add a new description if the input description not exists
        :param check_hint: (bool) whether to print check info
        :return: (tuple) the truly added pair (input words could be synonyms, return the origin pair), if failed, return None
        """
        if not self.have_target(target, True):
            if able_add_target:
                self.add_new_target(target, target_type, True)
            else:
                if check_hint:
                    print("failed to add sentiment pair (%s,%s,%s): %s is not a target" % (
                        target, description, sentiment, target))
                return None
        if able_overwrite or not self.have_target_description_pair(target, description):
            return self.modify_sentiment(target, description, sentiment, able_add_description=able_add_description,
                                         check_hint=check_hint)
        else:
            if check_hint:
                print(
                    "failed to add sentiment pair (%s,%s,%s): there is another pair in knowledge base: (%s,%s,%s)" % (
                        target, description, sentiment, target, description,
                        self.sentiment_of_target_description_pair(target, description, True)))
            return None

    def modify_sentiment(self, target, description, sentiment, able_add_description=True, check_hint=True):
        """
        modify the sentiment of pair (target,description), if description is not a description word for target, then add it
        :param target: (string) input target(synonym)
        :param description: (string) input description word
        :param sentiment: (string) input sentiment
        :param able_add_description: (bool) whether to add the input description to description set if it doesn't exists
        :param check_hint: (bool) whether to print check info
        :return: (tuple) the truly added pair (input words could be synonyms, return the origin pair), if failed, return None
        """
        # check target
        origin_target = self.target_of_synonym(target, False)
        if origin_target is None:
            if check_hint is True:
                print("failed to modify sentiment pair (%s,%s,%s): %s is not a target" % (
                    target, description, sentiment, target))
            return None
        # check description
        if not self.have_description(description):
            if not able_add_description:
                if check_hint:
                    print("failed to modify sentiment pair (%s,%s,%s): %s is not a description" % (
                        target, description, sentiment, description))
                return None
            self.add_new_description(description)
        # check sentiment
        if sentiment not in ["POS", "NEU", "NEG"]:
            if check_hint:
                print("failed to modify sentiment pair (%s,%s,%s): %s is not a sentiment" % (
                    target, description, sentiment, sentiment))
            return None

        descriptions = self.descriptions_of_target(target, False)
        if description not in descriptions:
            self.pairTD[origin_target].add(description)
            self.pairDT[description].add(origin_target)
        self.pairSentiment[(origin_target, description)] = sentiment
        return origin_target, description, sentiment

        # APIs for knowledge base modify ----end----


def knowledge_base_init(product='汽车'):
    print("Knowledge Base Initializing for product %s..." % product)
    config = KnowledgeBaseConfig(product)
    config.LOAD_FROM_CACHE = False
    if config.LOAD_FROM_CACHE and os.path.exists(config.CACHE_PATH):
        knowledge = KnowledgeBase.load_from_cache(config.CACHE_PATH)
        knowledge.write_js_variables(False)
    else:
        knowledge = KnowledgeBase()
        knowledge.load_knowledge_base(entity_word_dir=config.ENTITY_WORD_PATH,
                                      attribute_word_dir=config.ATTRIBUTE_WORD_PATH,
                                      description_word_dir=config.DESCRIPTION_WORD_PATH,
                                      whole_part_dir=config.WHOLE_PART_PATH,
                                      entity_attribute_dir=config.ENTITY_ATTRIBUTE_PATH,
                                      target_description_sentiment_dir=config.TARGET_DESCRIPTION_SENTIMENT_PATH,
                                      entity_synonym_dir=config.ENTITY_SYNONYM_PATH,
                                      attribute_synonym_dir=config.ATTRIBUTE_SYNONYM_PATH, product_name=product,
                                      js_cache_path=config.JS_CACHE_PATH,
                                      prob_da_dir=config.PROB_DA_PATH)
        knowledge.save(config.CACHE_PATH)
    print("Knowledge Base Initializing Done")
    return knowledge
