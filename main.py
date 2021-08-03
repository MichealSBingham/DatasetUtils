#!/usr/bin/env python3

"""
Created on Wed Jun 16 14:17:07 2021

@author: Ruth Lee & Rishyak Panchal
encoding: utf-8

script to rturn each person object and read in .tsv file
"""

import os
import csv

from Person import Person
from Lover import Lover
# Person(name, *, birthday=None, birthplace='nyc', gender=None,
#	sun=None, moon=None, mercury=None, venus=None, mars=None,
#	def_love=None, ideal_lover=None, lovers=[])
import attraction_model

class Model:
    """
    Model Object
    Holds all required functions to run it.
    """

    def __init__(self, tsv_file="responses.tsv", first_line=True):
        """
        Constructor.

        tsv_file -- link to the config file to retrieve global args & kwargs
        first_line -- True if header else False

        sig: string, bool -> NoneType
        """
        self.tsv_file = tsv_file
        self.first_line = first_line
        self.list = self.make_list(self.config.tsv_file)

    def make_list(self, tsv):
        """
        Makes the list that the model runs.

        tsv -- name of input TSV file.

        sig: string -> list
        """
        
        def string_check(answer):
            """
            Verifies the string input.

            answer -- The name of the cache. "L1" or "L2"

            sig: string -> bool
            """
            punct = set('@^! #%$&)(+*-="')
            new_list = answer.split(", ")
            return not any(char in punct for item in new_list for char in item)

        # making bday format into month/day/year
        def birthday_form(birthday):
            """
            Converts plain text birthday input to MM/DD/YYYY.

            birthday -- birthday in plain text

            sig: string -> string
            """
            year = birthday[0:4]  ## TODO: Can be made more robust than string slicing and avoid whitespace errors.
            month = birthday[5:7]
            day = birthday[8:10]
            birthday = f"{month}/{day}/{year}"
            return birthday
        
        tsv_file = open(tsv)
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
                bday = birthday_form(bday)
                
                # isCorrect = string_check(def_love) #check format of thei what is love answer
                # if isCorrect == False:
                #     def_love = None
                # else:
                #     def_love = def_love.split(",")
                def_love = def_love.split(", ") if string_check(def_love) else None

                # isCorrect = string_check(ideal_lover)
                # if isCorrect == False:
                #     ideal_lover = None
                # else:
                #     ideal_lover = ideal_lover.split(",")
                ideal_lover = ideal_lover.split(", ") if string_check(ideal_lover) else None

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
                        loverBday = birthday_form(loverBday)
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

    def train(self):
        attractiveness_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())

        checkpointer = ModelCheckpoint(filepath='attractiveness.hdf5', monitor = "val_loss", verbose=1, save_best_only=True, mode = 'auto')

        earlyStop = EarlyStopping(monitor='val_loss', patience=50)

        score = attractiveness_model.fit(train_x, train_y, epochs=5000, validation_data=(val_x, val_y), callbacks=[checkpointer, earlyStop])

    def run(self):
        pass



if __name__ == "__main__":
    model = Model()

    # model.train()
    # model.run()