# -*- coding:utf-8 -*-

def split_sent(text_, spliter="。？?！"):
    list_sents = []
    tmp_sent = ""
    for char_ in text_:
        if char_ in spliter:
            if len(tmp_sent) == 0:
                continue
            else:
                tmp_sent += char_
                list_sents.append(tmp_sent)
                tmp_sent = ""
    if len(tmp_sent) > 0:
        list_sents.append(tmp_sent)
    return list_sents
