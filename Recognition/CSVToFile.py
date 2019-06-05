import csv
import numpy as np
def CSVToFile(data,filepath,type=0,isOneLine=False):         #type=0代表传入的数据不包含之前的数据，type=1表示传入的数据包含了之前的数据,默认为0
    '''
    type
    0 追加写入（默认）
    1 覆盖写入
    '''
    if type ==0:
        try:
            with open(filepath, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                olddata = []
                for row in reader:
                    olddata.append(row)
        except FileNotFoundError:
            olddata=[]
            pass
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            for row in olddata +data:
                writer.writerow(row)

        return True
    else:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            for row in data:
                writer.writerow(row)

        return True


def cleanFile(filepath):
    with open(filepath,'w',encoding='utf-8') as f:
        f.truncate()
    return True
