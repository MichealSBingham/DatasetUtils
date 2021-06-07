from flatlib.datetime import Datetime
from flatlib.chart import Chart
from flatlib.geopos import GeoPos


SIGNS = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
ASPECTS = ['Square', 'Conjunct', 'Semisextile', 'Sextile', 'Trine', 'Quincunx', 'Opposition']

#birthday: Datetime object . Date of birth 
#city [optional]: string - default is 'nyc'. Options are nyc,atl
#Returns the sign that each of the planet is in given birthday and city 
def getPlanets(birthday, city='nyc'):

	date = birthday

	#Creating GeoPos object from city needed for chart 
	if city == 'nyc':
		pos = GeoPos(40.7128, -74.0060)
	elif city == 'atl':
		pos = GeoPos(33.7490, -84.3880)
	elif city == 'sf':
		pos = GeoPos(37.7749, -122.4194)
	else:
		pos = GeoPos(40.7128, -74.0060)

	chart = Chart(date, pos)

	sun = chart.get('Sun')
	moon = chart.get('Moon')
	mercury = chart.get('Mercury')
	venus = chart.get('Venus')
	mars = chart.get('Mars')

	return {'sun': sun.sign, 'moon': moon.sign, 'mercury': mercury.sign, 'venus': venus.sign, 'mars': mars.sign}




#Returns the angle of the sign of the zodiac [Aries 30 deg, etc]
def getAngle(sign):
	return (SIGNS.index(sign)+1)*30


#Gets  Aspect between 2 signs 

def getAspect(sign1, sign2):
	theta = getAngle(sign1) #in degrees what the sign is on the zodiac  , angle 1 
	phi = getAngle(sign2) #in degrees what the sign is on the zodiac, angle 2 
	angle = abs(theta-phi)

	if angle > 180:
		angle = 360 - angle 

	if angle == 90:
		return 'Square'
	elif angle == 0:
		return 'Conjunct'
	elif angle == 30:
		return 'Semisextile'
	elif angle == 60:
		return 'Sextile'
	elif angle == 120:
		return 'Trine'
	elif angle == 150:
		return 'Quincunx'
	elif angle == 180:
		return 'Opposition'
	else: 
		return 



