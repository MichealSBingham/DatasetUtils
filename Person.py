from chart import * 

class Person:



#Provide a name and birthday when you instantiate a person object. 
#Their birth chart (sun, moon, etc attributes) will automatically be computed unless you explicitly set them after instantiating the Person object


# name - String
# birthday - DateTime Object representing birthday OR you can enter a string 'YYYY/MM/DD' and it'll convert to DateTime
# birthplace [optional*]: except in the cases where the person was born outside of the US, we automatically use nyc for the birthplace. This is because for our calculations the birthplace won't cause too much of a difference unless they are outside of the country
#In the case, adjust accordingly below because I have not added functionality to this yet 
# sun - String. One of 12 of the zodiac signs
# moon ..
# ...
# mars - String. One of 12 of the zodiac signs. "Capricorn, Cancer, etc."
#def_love: List of strings. Words the person used to define love. 
#ideal_lover: List of strings. Words the person used to describe their ideal lover
#Lovers: List of Lover objects. Prior romantic encounters 
	def __init__(self, name, *, birthday=None, birthplace='nyc', 
				 sun=None, moon=None, mercury=None, venus=None, mars=None, 
				 def_love=None, ideal_lover=None, lovers=[]):

		didNotGiveBirthday = (birthday == None)
		didNotGiveChart = (sun == None or moon == None or mercury == None or venus == None or mars == None)

		if didNotGiveBirthday and didNotGiveChart:
			raise ValueError("Invalid Parameters: You must either provide the birthday or specify what signs their planets are in.")

		self.name = name
		if type(birthday) == Datetime:
			self.birthday = birthday
		else: #make datetime object 
			self.birthday = Datetime(birthday)
		self.birthplace = birthplace

		sign = getPlanets(self.birthday, birthplace)


		self.sun = sign['sun'] 
		self.moon = sign['moon']  
		self.mercury = sign['mercury'] 
		self.venus = sign['venus'] 
		self.mars = sign['mars'] 
		self.def_love = def_love
		self.ideal_lover = ideal_lover
		self.lovers = lovers



	## Adds a lover to the Person and assigns the lover to the person.
	def add_lover(self, lover):
		lover.person = self
		self.lovers.append(lover)

	def get_num_lovers(self):
		return len(self.lovers)



