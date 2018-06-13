import os

path = './dataset'
files = os.listdir(path)
png_name = 0
posi_01 = 17
posi_02 = 19
posi_03 = 21
posi_04 = 23
posi_05 = 25
posi_06 = 27
posi_07 = 29
posi_08 = 31
posi_09 = 33
posi_10 = 35
posi_11 = 37
posi_12 = 39
posi_13 = 41
posi_14 = 43
posi_15 = 44

for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        f = open(path + "/" + file, 'r')  # 打开文件
        f_class_1 = open(path + '/' + 'class_1_' + file, 'w')
        f_class_2 = open(path + '/' + 'class_2_' + file, 'w')
        f_class_3 = open(path + '/' + 'class_3_' + file, 'w')
        f_class_4 = open(path + '/' + 'class_4_' + file, 'w')
        iter_f = iter(f)  # 创建迭代器
        str_temp = ''
        for line in iter_f:
            class_1_str = line[png_name:posi_01] + line[posi_01:posi_02] + line[posi_03:posi_04] + \
                          line[posi_04:posi_05] + line[posi_08:posi_09] + line[posi_10:posi_11] + \
                          line[posi_13:posi_14] + '\n'
            f_class_1.write(class_1_str)

            class_2_str = line[png_name:posi_01] + line[posi_05:posi_06] + line[posi_06:posi_07] + \
                          line[posi_09:posi_10] + line[posi_14:posi_15] + '\n'
            f_class_2.write(class_2_str)

            class_3_str = line[png_name:posi_01] + line[posi_02:posi_03] + line[posi_06:posi_07] + '\n'
            f_class_3.write(class_3_str)

            class_4_str = line[png_name:posi_01] + line[posi_07:posi_08] + line[posi_12:posi_13] + '\n'
            f_class_4.write(class_4_str)
        f.close()
        f_class_1.close()
        f_class_2.close()
        f_class_3.close()
        f_class_4.close()
