import json
WEIGHT_DIC = dict()
ATTRIBUTES = ("sku",
              "designation",
              "price",
              "msrp",
              "characteristics",
              "fortified",
              "description",
              "image",
              "date_added",
              "single_product_url",
              "is_vintage_wine",
              "drink_from",
              "drink_to",
              "is_blend",
              "name",
              "acidity",
              "body",
              "tannin",
              "alcohol_pct",
              "sweetness",
              "varietals",
              "popularity",
              "purpose",
              "flaw",
              "vintage",
              "qpr",
              "styles",
              "bottle_size",
              "highlights",
              "color_intensity",
              "discount_pct",
              "foods",
              "rating",
              "product_id",
              "qoh",
              "wine_type",
              "region",
              "brand",
              "prototype",
              "short_desc")

for weight in ATTRIBUTES:
    WEIGHT_DIC[weight.strip()] = 1.0

def scoring(structured_data):
    summary_list = []
    text_list = structured_data['text']
    attribute_list = structured_data['attribute']
    summary_dic = []
    for i,text in enumerate(text_list):
        score = 0
        attribute_count = 0
        sentence_attribute = []
        attributes = attribute_list[i][0][0]
        if attributes['attributes'] is None:
            continue
        for attribute in attributes['attributes']:
            if attribute['code'] in WEIGHT_DIC.keys():
                weight = WEIGHT_DIC[attribute['code']]
            else:

                weight = 1.0
            sentence_attribute.append(attribute)
            score += weight
            attribute_count += 1
        if attribute_count != 0:
            score /= attribute_count
        sentence_dic = dict()
        sentence_dic['index'] = i
        sentence_dic['text'] = text
        sentence_dic['score'] = score
        sentence_dic['attribute'] = sentence_attribute
        summary_dic.append(sentence_dic)
    summary_list.append(
        sorted(summary_dic, key=lambda x: x['score'], reverse=True))
    return summary_list

def adopt(sentences_dic, max_words_len):
    words_leng = 0
    summary = []
    for sentence_dic in sentences_dic:
        if words_leng + len(sentence_dic['text'].split(' ')) < max_words_len:
            summary.append(sentence_dic)
            words_leng += len(sentence_dic['text'].split(' '))
    summary_dic = dict()
    summary_dic['summary'] = sorted(summary, key=lambda x: x['index'],
                                    reverse=False)
    summary_dic['summary_leng'] = words_leng
    return summary_dic

def extract_text(sorted_sum_dic):
    summary_text = []
    score = 0.0
    attributes = []
    for summary in sorted_sum_dic['summary']:
        summary_text.append(summary['text'].strip())
        score += summary['score']
        attributes.append(summary['attribute'])

    return summary_text, score, attributes

def summarizate(structured_data,max_words_len):
    sum_dic_list=scoring(structured_data)
    summary_list = []
    for sum_dic in sum_dic_list:
        text, score,attributes = extract_text(adopt(sum_dic, max_words_len))
        summary_dic = dict()
        summary_dic['text'] = text
        summary_dic['score'] = score
        summary_dic['attribute'] = attributes
        summary_list.append(summary_dic)
    summary_list = sorted(summary_list, key=lambda x: x['score'],
                          reverse=True)  # get the highest score summary
    best_summary = summary_list[0]

    return best_summary


file = open('attributes_review_.txt', 'r', encoding='utf-8')
sentences_file = open('sml_summary.txt','w',encoding='utf-8')
for i, review in enumerate(file):
    if i>=2000:
        break
    review = json.loads(review)
    best_summary = summarizate(review, 45)
    sentences_file.write(json.dumps(best_summary)+'\n')
sentences_file.close()