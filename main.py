#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:17:07 2021

@author: Ruth Lee
"""
#script to rturn each person object and read in .tsv file
from Person import Person
import csv
from Lover import Lover
#(self, name, *, birthday=None, birthplace='nyc', gender=None,
#				 sun=None, moon=None, mercury=None, venus=None, mars=None,
#				 def_love=None, ideal_lover=None, lovers=[])


def main():
    tsv_file = open("responses.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    pList = []
    firstLine = True
    for row in read_tsv:
        if firstLine == False:
            bday = str(row[1])  # accessing bday from .tsv
            name = row[2]  # accessing name from .tsv
            gender = row[5]
            def_love = row[6]
            ideal_lover = row[7]
            bday = birthdayForm(bday)
            
            isCorrect = string_check(def_love) #check format of thei what is love answer
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
            addAnother = 1
            x = 8
            while addAnother == 1:
                loverName = row[x]
                #loverGender = row[x+1]
                if row[x+2] == "I know their birthday":
                    loverBday = str(row[x+9])
                    loverBday = birthdayForm(loverBday)
                    sun=None
                    moon=None
                    mercury=None
                    venus=None
                    mars=None
                else:
                    #if the bday is unknown, need to obtain star sign data
                    sun = row[x+3]
                    moon = row[x+4]
                    mercury = row[x+5]
                    venus = row[x+6]
                    mars = row[x+7]
                    #zodiac sign things
                attractiveness = row[x+10]
                chemistry = row[x+11]
                sex = row[x+12]
                love = row[x+13]
                harmony = row[x+14]
                duration = row[x+15]
                isCurrent = row[x+16]
                relationship_type = row[x+17]
                #covert data to the relationship_type keywords
                if relationship_type == "Better Off Friends":
                    relationship_type = "BOF"
                elif relationship_type == "Friends With Benefits":
                    relationship_type = "FWB"
                elif relationship_type == "Fling":
                    relationship_type = "FLING"
                elif relationship_type == "Partner":
                    relationship_type = "P"
                elif relationship_type == "Soulmate":
                    relationship_type = "S"
                #make lover object and add it to person
                #print(loverName, loverBday, sun, moon,mercury, venus, isCurrent,attractiveness, sex, love, chemistry, harmony, relationship_type)
                lover = Lover(loverName, birthday=loverBday, sun=sun, moon=moon, mercury=mercury, venus=venus, mars=mars, duration=duration, 
				 isCurrent=isCurrent,
				 attractiveness=attractiveness, 
				 chemistry=chemistry, 
				 sex=sex, 
				 love=love, 
				 harmony=harmony,
				 relationship_type=relationship_type, 
				 person = p)
                p.add_lover(lover)
                addAnother = int(row[x+18])
                x = x + 19
                
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

def birthdayForm(birthday):
    #making bday format into month/day/year
    year = birthday[0:4]
    month = birthday[5:7]
    day = birthday[8:10]
    birthday = month + "/" + day + "/" + year
    return birthday
if __name__ == "__main__":
    main()











