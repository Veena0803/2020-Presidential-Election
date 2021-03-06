# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:54:31 2020

@author: nkvar
"""

"""
Dictionary of pollster name changes, going from names as 
written on 270towin.com to names as written on Project 538

This will let you get grades/biases for around 85% of the polls.
For remaining polls, consider that information to be missing data.

Tip:  from pollster_name_changes import pollster_changes

Then loop over the poll names in the surveys you collected, look them up 
in the dictionary, and make a name if the pollster is in the keys.
"""

pollster_changes = {'CNBC/Change Research' : 'Change Research',
                    'Public Policy' : 'Public Policy Polling',
                    'Axios / SurveyMonkey' : 'SurveyMonkey',
                    'NY Times / Siena College' : 'Siena College/The New York Times Upshot',
                    'Quinnipiac' : 'Quinnipiac University',
                    'YouGov/CBS News' : 'YouGov',
                    'Ipsos/Reuters' : 'Ipsos',
                    'Fox News' : 'Fox News/Beacon Research/Shaw & Co. Research',
                    'Baldwin Wallace Univ.' : 'Baldwin Wallace University',
                    'NBC News/Marist' : 'Marist College',
                    'Mason-Dixon' : 'Mason-Dixon Polling & Strategy',
                    'CNN/SSRS' : 'CNN/Opinion Research Corp.',
                    'Marquette Law' : 'Marquette University Law School',
                    'East Carolina Univ.' : 'East Carolina University',
                    'Benenson / GS Strategy' : 'Benenson Strategy Group',
                    'Florida Atlantic Univ.' : 'Florida Atlantic University',
                    'Rasmussen Reports' : 'Rasmussen Reports/Pulse Opinion Research',
                    'UT Tyler' : 'University of Texas at Tyler',
                    'VCU' : 'Virginia Commonwealth University',
                    'Landmark Comm.' : 'Landmark Communications',
                    'UMass Lowell' : 'University of Massachusetts Lowell',
                    'Remington Research' : 'Remington Research Group',
                    'Susquehanna' : 'Susquehanna Polling & Research Inc.',
                    'WPA Intelligence' : 'WPA Intelligence (WPAi)',
                    '1892' : '1892 Polling',
                    'ABC News / Wash. Post' : 'ABC News/The Washington Post',
                    'Christopher Newport Univ.' : 'Christopher Newport University',
                    'East Tennessee State' : 'East Tennessee State University',
                    'Emer' : 'Emerson College',
                    'Fabrizio Lee' : 'Fabrizio, Lee & Associates',
                    'Fairleigh Dickinson' : 'Fairleigh Dickinson University (PublicMind)',
                    'Franklin & Marshall' : 'Franklin & Marshall College',
                    'GQR Research' : 'GQR Research (GQRR)',
                    'Garin-Hart-Yang' : 'Garin-Hart-Yang Research Group',
                    'Gonzales Research' : 'Gonzales Research & Marketing Strategies Inc.',
                    'HighGround' : 'HighGround Inc.',
                    'InsiderAdvantage' : 'Opinion Savvy/InsiderAdvantage',
                    'MRG' : 'MRG Research',
                    'MassINC' : 'MassINC Polling Group',
                    'Meeting Street Insights' : 'Meeting Street Research',
                    'Mitchell Research' : 'Mitchell Research & Communications',
                    'Montana State U.' : 'Montana State University Billings',
                    'Morning Call / Muhlenberg' : 'Muhlenberg College',
                    'PPIC' : 'Public Policy Institute of California',
                    'Research America' : 'Research America Inc.',
                    'Rutgers-Eagleton' : 'Rutgers University',
                    'SLU/YouGov' : 'Saint Leo University',
                    'Selzer & Company' : 'Selzer & Co.',
                    'Sooner Poll' : 'SoonerPoll.com',
                    'Sooner Survey' : 'SoonerPoll.com',
                    'St. Leo University' : 'Saint Leo University',
                    'THPF/Rice Univ.' : 'Rice University',
                    'TargetSmart' : 'TargetSmart/William & Mary',
                    'UC Berkeley' : 'University of California, Berkeley',
                    'UMass Amherst/WCVB' : 'University of Massachusetts Amherst',
                    'USC' : 'USC Dornsife/Los Angeles Times',
                    'Univ. of Colorado' : 'University of Colorado',
                    'Univ. of New Hampshire' : 'University of New Hampshire',
                    'Univ. of Georgia' : 'University of Georgia',
                    'Univ. of North Florida' : 'University of North Florida',
                    'Univ. of Texas / Texas Tribune' : 'University of Texas at Tyler',
                    'Univ. of Wisconsin-Madison' : 'University of Wisconsin (Badger Poll)',
                    'Univision/ASU' : 'Arizona State University',
                    'Univision/CMAS UH' : 'Univision/University of Houston/Latino Decisions',
                    'Yahoo/YouGov': 'YouGov',
                    'YouGov/CBS News': 'YouGov'
                    }