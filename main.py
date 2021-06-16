#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:17:07 2021

@author: Ruth Lee
"""
#script to rturn each person object and read in .tsv file
from Person import Person
import csv
#(self, name, *, birthday=None, birthplace='nyc', gender=None,
#				 sun=None, moon=None, mercury=None, venus=None, mars=None,
#				 def_love=None, ideal_lover=None, lovers=[])


def main():
    tsv_file = open("Relationship Dataset - Relationship Research.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    pList = []
    firstLine = True
    for row in read_tsv:
        if firstLine == False:
            bday = str(row[0])  # accessing bday from .tsv
            name = row[1]  # accessing name from .tsv
            gender = row[4]
            def_love = row[5]
            ideal_lover = row[16]

            isCorrect = string_check(def_love)
            if isCorrect == False:
                def_love = None
            else:
                def_love = def_love.split(",")

            isCorrect = string_check(ideal_lover)
            if isCorrect == False:
                ideal_lover = None
            else:
                ideal_lover = ideal_lover.split(",")

            p = Person(name, birthday=bday, gender=gender, def_love=def_love,
                       ideal_lover=ideal_lover)  # make person object based on data
            #still need to add lovers
            pList.append(p)
        else:
            firstLine = False
    tsv_file.close()
    return pList


# Function to check the string
def string_check(answer):
    punct = set('@^! #%$&)(+*-="')
    new_list = answer.split(", ")
    return not any(char in punct for item in new_list for char in item)


if __name__ == "__main__":
    main()
