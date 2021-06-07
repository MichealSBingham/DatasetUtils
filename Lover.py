from Person import Person 

#BOF: better off friends, FWB: Friends with benefits, 
#FLING: Fling, P: Partner, S: Soulmate
RELATIONSHIP_TYPES = ['BOF', 'FWB', 'FLING', 'P', 'S']


class Lover(Person):
	"""The lover is the other romantic partner in the relationship. 
	Example if you collect romantic data between Micheal and Valentine and Micheal is 
	the one reporting the data,
	Micheal is the Person and 
	Valentine is the Lover.
	A Person object can have many Lovers (if they reported more than one).

	It inherits from a Person object except def_love, ideal_lover, and lovers should be None. (because we did not collect data on this from them! Data was collected from 'Person' concerning the 'Lover'. We never spoke to the 'Lover'

	duration: Time of relationship in months [required param]
	attractiveness: 1-10 Integer or double on how attractive 'Person' found the 'Lover'
	chemistry: 1-10 Integer on how much chemistry existed  
	sex: 1-10 Integer on how great the sex was
	love: 1-10 Integer on how much  love "Person" had for the "Lover"
	relationship_type : One of RELATIONSHIP_TYPES
	person: A 'Lover' object can have only 1 'Person' associated with it. (see above) 
	"""

	def __init__(self, name, *, birthday=None, birthplace='nyc', sun=None, moon=None, mercury=None, venus=None, mars=None, 
				 def_love=None, ideal_lover=None, lovers=None,
				 duration=None, 
				 isCurrent=None,
				 attractiveness=None, 
				 chemistry=None, 
				 sex=None, 
				 love=None, 
				 harmony=None,
				 relationship_type=None, 
				 person = None):


		self.duration = duration
		self.isCurrent = isCurrent
		self.attractiveness = attractiveness
		self.chemistry = chemistry
		self.sex = sex 
		self.love = love
		self.harmony = harmony
		self.relationship_type = relationship_type
		self.person = person

		Person.__init__(self, name, birthday=birthday, birthplace=birthplace)


		
		
