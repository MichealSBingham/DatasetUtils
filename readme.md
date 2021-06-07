




# How to parse relationship survey data (v1 Typeform)

## Documentation 


This is Best explained by example: 

### `Person `

A `Person` object 

```
   Person(self, 
   name, *, 
   birthday=None, 
   birthplace='nyc', 
	sun=None, moon=None, mercury=None, venus=None, mars=None, 
	def_love=None, 
   ideal_lover=None, 
    lovers=[])
   ```

Each datry entry (row) will give information on a person and their prior lovers/relationships. 
For instance if Micheal completes the survey form and he provides data on his past relationships with 
Gracen and Valerie...

Then Micheal is a `Person` object. 
Gracen and Valerie are `Lover` objects. 
 
 
 
 
 
 
 
 
 
 
 

###### Creating a `Person` (Instantiating)  

A `Person` object must be instantiated with the name *and* either the person's birthday or *all* of the (sun, moon, mercury, venus, mars) parameters such as 

```
from chart import * 
from Person import Person 
from Lover import Lover 
from Relation import Relation 


micheal = Person('Micheal', birthday='1999/07/21')
micheal = Person('Micheal', sun='Cancer', moon='Scorpio', mercury='Leo', venus='Virgo', mars='Scorpio')

```






_Note_

 The `birthday` parameter can accept: 
      a `string` with  `'YYYY/MM/DD'` format or 
		a `Datetime` object that represents the date. See `Datetime` documentation (Google) 
					ex: `date = Datetime('1999/07/21')`. May be (or may not be) easier to deal with Datetime module since you'll be reading it from a spreadsheet and I know sometimes they have a weird way of formatting dates 


The only time you would do this this way is if a `Person` (or a `Lover`) does not have it's birthday available but has the birthchart.
There will be a column in each row that has an entry for either the birthday or the full birth chart so there won't be any issues there





The attributes of a `Person` are mutable and can be changed by reference

```
micheal.birthplace = "nyc" #Redundant parameter for now. Only going to be useful in the feature. The constructor will set the default value here anyway
so don't worry about this.
 
micheal.def_love = ['unconditional', 'genuine', 'forever'] #A list of adjectives Micheal (A 'Person') used to describe love 

micheal.ideal_lover = ['sensitive', 'beautiful', 'fierce'] #A list of adjectives  the person used to describe their ideal lover

```


 You can also read and write to the following properties; however, these properties are 
 automatically set IF you include the birthday parameter in the constructor. 


You will **only need** to write to these if you do not have the person's birthday but you have this information. Which is possible, because some data entries have columns for the birth chart but not the birthday 
 
 ```
 micheal.moon = 'Scorpio'
 micheal.sun = 'Cancer'
 micheal.mercury = 'Leo'
 micheal.mars = 'Scorpio'
 micheal.venus = 'Virgo'
```


### `Lover `


 Finally, a  `Lover` is a past romantic encounter of a Person (Micheal). 

 A person can have many Lovers. So the `.lovers` attribute is an array of lovers but this is READ-ONLY

 A `Lover` inherits from the Person class 
 
 ```
 gracen = Lover('Gracen', '12/08/1998')
 valerie = Lover('Valerie', sun='Aries', moon='Aquarius', mercury='Gemini', venus='Gemini', mars ='Aries')
```

 ###### Adding a Lover to a Person 

 You can add Gracen and Valerie as lovers to Micheal with the `.add_Lover()` function

```
 micheal.add_lover(gracen)
 micheal.add_lover(valerie)
```


 Now the .lovers attribute will return an array with 2 objects, gracen and valerie. 

 ###### Lover Attributes (Read and Write)
 
 

 Examples: 

```
 gracen.duration = 1 # Months the relationship between Gracen and gracen.person (Micheal) lasted 
 gracen.isCurrent = False #boolean of whether or not this is a current partner ornot
 gracen.attractivness = 10 #gracen.partner's (Micheal's) attractiveness rating of gracen. Similarly as follows for chemistry, sex, and love
 gracen.chemistry = 9
 gracen.sex = 8 
 gracen.love = 6
 gracen.harmony = 9 
 gracen.relationship_type = 'FLING' #String, has to be one of the values in the RELATIONSHIP_TYPES list constant 
 gracen.person # Read only . This will automatically be assigned when you call micheal.add_lover(gracen)
```

### `Relation `


 Finally we have a `Relation` object. 

 A `relation` is an object associated with a single relationship between a person and a lover 

 Instantiate it with a Person and Lover object

```
 couple1 = Relation(micheal, gracen)
 couple2 = Relation(micheal, valerie)
```

 A `Relation` object has **READ-ONLY** properties that will contain all relationship information such as 
 
 ```
 couple1.partner # --> micheal object returned 
 couple1.lover # -> gracen object returned
 couple1.duration # -> returns 1 (in this specific example) returns the duration of relationship in months
 couple1.attractiveness, couple1.chemistry, ... , couple1.harmony 
 couple1.def_love
 couple1.sun_aspect # 'Quincunx' -> Returns one of the ASPECTS constants 

```

