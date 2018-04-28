#!/usr/bin/python
# -*- coding: utf-8 -*-


def entities2sentiments_single(entities):
    """根据entity树得到最终结果"""
    sentiments = []
    for enti in entities:
        for atri in enti.attributes:
            if len(atri.good_comments) != 0:
                for comment in atri.good_comments:
                    sentiments.append([enti.name, atri.name, comment, 1]);
            if len(atri.bad_comments) != 0:
                for comment in atri.bad_comments:
                    sentiments.append([enti.name, atri.name, comment, -1]);
            if len(atri.normal_comments) != 0:
                for comment in atri.normal_comments:
                    sentiments.append([enti.name, atri.name, comment, 0]);
    return sentiments


def entities2sentiments_group(entities):
    """根据entity树得到最终结果"""
    sentiments = []
    for enti in entities:
        for atri in enti.attributes:
            if len(atri.good_comments) or \
                    len(atri.bad_comments) or \
                            len(atri.normal_comments) != 0:
                sentiments.append(
                    [enti.name, atri.name, len(atri.good_comments), len(atri.bad_comments), len(atri.normal_comments)]);
    return sentiments
