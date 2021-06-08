
from chart import * 

class Relation():
	"""A Relation Object is a single relationship between a Person and a Lover. 
	You only need to pass person and lover into this constructor 

	A relation contains all the details of a relationship between a partner and lover
"""
	def __init__(self,
				 person, 
				 lover):

		self.person = person
		self.lover = lover
		self.attractiveness = lover.attractiveness
		self.chemistry = lover.chemistry
		self.sex = lover.sex
		self.love = lover.love
		self.harmony = lover.harmony
		self.type = lover.relationship_type
		self.duration = lover.duration

		self.sun_aspect = getAspect(person.sun, lover.sun)
		self.moon_aspect = getAspect(person.moon, lover.moon)
		self.mercury_aspect = getAspect(person.mercury, lover.mercury)
		self.venus_aspect = getAspect(person.venus, lover.venus)
		self.mars_aspect = getAspect(person.mars, lover.mars)

		self.def_love = person.def_love
		self.ideal_lover = person.ideal_lover

	"""
	Class method 
	- returns all of the relation objects of a particular person in a list 

	"""
	def __create_relations__(person):
		relations = []
		for lover in person.lovers:
			relation = Relation(person, lover)
			relations.append(relation)

		return relations






	
		