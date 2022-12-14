# Bahn-Vorhersage [https://bahnvorhersage.de/](https://bahnvorhersage.de/)

```bash
████████████████████████████████████▇▆▅▃▁
       Bahn-Vorhersage      ███████▙  ▜██▆▁
███████████████████████████████████████████▃
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█████▄▖
█████████████████████████████████████████████
 ▜█▀▀▜█▘                       ▜█▀▀▜█▘   ▀▀▀
```

Bahn-Vorhersage (formerly known as TrAIn_Connection_Prediction) is a train delay prediction system for German railways. We are trying to make rail travel more user-friendly. Our machine learning system predicts the probability that all connecting trains in a connection will be coughed.

## History
| Date                        | Event                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Summer 2019                 | We were searching for an AI project in order to compete in the German National Artificial Intelligence Competition 2019 called [BWKI](https://bw-ki.de) - TrAIn_Connection_Prediction was born                                                                                                                                                                                                                                                                                                                                                                                                                        |
| November 30th, 2019         | We won the BWKI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| January 2020                | Started to switch our data source from https://zugfinder.de to IRIS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| February 4th, 2020          | We were invited by DB Analytics to present our project                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| February 18th 2020          | We won the regional contest of Jugend Forscht. All further competitions this year were canceled due to covid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| March 2020                  | After we moved all the data we don't own out of our git-repository, we finally open sourced our project                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Also March 2020             | In order to improve prediction accuracy, we started to gather information about the railway network                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Spring and Summer 2020      | Using IRIS as data source and finding useful data for the railway network is way harder than expected. IRIS is not made at all the gather historic delays and needs to be fetched every 2 Minutes for every single station. Network data is available in some formats, but not in a routable one. Trassenfinder API of DB Netz does not contain private rail tracks. BTW, did you even know that there is a unit called hecto meter (1 hm = 100 m)? At least Trassenfinder API uses it. In the end we wrote a script to parse stations into a railway network gathered with the help of OSMnx.                        |
| September 2020              | It is not only hard to get the IRIS data in the first place, but after gathering data for some time, parsing the data gets very painful. At this time, our database server (4 cores, HDD) takes up to 5 days in order to parse the data.                                                                                                                                                                                                                                                                                                                                                                              |
| October 2020                | Switched from random forests to xgboost                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| December 2020               | We moved our servers from the Kepler-Gymnasium to the SFZ Eningen                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| January 2021                | As we are participating at Jugend Forscht once again, we are writing new docs for the project and also improving some things in the last second. Our web server is now much easier to deploy, and we actually have some stats for our now hyperparameter tuned machine learning models.                                                                                                                                                                                                                                                                                                                               |
| February 2021               | Just before Jugend Forscht, we decided to change our frontend to use VueJS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| February 26th 2021          | We once again win the local Jugend Forscht competition, so we qualify for the state competition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| March 2021                  | We managed to collect some donations and buy a powerful server / workstation for that. We put a Kubernetes cluster on that server and secured our new domain with let's encrypt. Parsing the data now takes one or two hours instead of a week. We also started to crawl obstacle / construction works data.                                                                                                                                                                                                                                                                                                          |
| March 24th 2021             | We won the 3rd prize at the Jugend Forscht state competition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Spring and Summer 2021      | A lot was changed but not a lot was changed: During the competition we always had to match deadlines, so many things were left in a kind of working, but not automated nor robust way. We did a lot in order to make the code cleaner and more robust. Also, the Deutsche Bahn does not want us as affiliate partner, so our website will probably not generate any income any time soon                                                                                                                                                                                                                              |
| October 2021                | As we plan to have some kind of arrival and departure boards at some point in time, we rewrote the whole data parser in order to parse gathered data within a few seconds. This was very hard as our parser relied heavily on caching, which relied on the data being sorted by station. In order to make the real-time parsing work, we had to parse the data unordered. To solve this, we added a persistent cache, that could be stored in our database.                                                                                                                                                           |
| January 2022                | The frontend has grown over the years, and it was time to clean it up. Instead of style classes for every button and text box, we finally made proper use of bootstrap and created our own theme                                                                                                                                                                                                                                                                                                                                                                                                                      |
| February 2022               | We rebranded our project as Bahn-Vorhersage instead of the hard to memorize "TrAIn_Connection_Prediction". By doing that, we also moved our frontend out of the repository and on GitLab                                                                                                                                                                                                                                                                                                                                                                                                                              |
| December 2021 to March 2022 | We realized that station names etc. change over time. This meant that we needed to introduce a new way of modeling stations, as we do not only need the current station information but also all the historic data. Otherwise, we could not parse the data we gathered one year ago properly. This also meant changing everything that relies on that station data (almost the whole project) and coding some kind of station update utility. This was also way harder than expected, as there are about 10 places to search for station data but none of them is complete                                            |
| March 2022                  | Also moved the rest of the code to GitLab                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| May 2022                    | Beginning on the first of June, Germany introduced the 9€-Ticket for local trains. So we needed a way to route for local trains only. That was easy, as Marudor's API has support for that. However, we found that if we added more options to our search, we should also add an option to search for routes with bikes. And that is not supported by Marudor. So we switched our routing provider once again and moved to a self-hosted instance of https://v5.db.transport.rest/. For that, we had to rewrite all of our server parsing and rendering code. At least now, we use the Friendly-Public-Transit-Format |
| August 2022                 | Finally, automatically train ML-Models every night to have up-to-date models in the morning. As always, this was way harder than thought, because Kubernetes does not support GPU out of the box. And microk8s runs on containerd instead of docker, which makes it event harder to run gpu because the k8s backend has to be switched from conternerd to docker to nvidia-docker                                                                                                                                                                                                                                     |
| August 2022                 | Trying to get into the affiliate partner program from DB, we added fancy cards for travel destinations in order to cause desire to travel. This made us rewrite almost all routing code for the frontend and allowed us to introduce a share button to share connection searches. What didn't happen was that we were excluded from the partner program, due to tactical reasons                                                                                                                                                                                                                                      |
| October 2022                | It's restyle time! After we found out that DB stole our naming scheme for beta test sites (https://next.bahn.de/ mimics our beta testing site https://next.bahnvorhersage.de/) we inspired ourselves a little bit and remade our design to be way more clear and more intuitive                                                                                                                                                                                                                                                                                                                                       |


## Further Information

For the youth competition [Jugend Forscht](https://www.jugend-forscht.de/) (JuFo) in Germany we have written a paper (in german) about our project,  
which can be found in our repository under [docs/langfassung_tcp.pdf](https://gitlab.com/bahnvorhersage/bahnvorhersage/-/blob/master/docs/langfassung_tcp.pdf).

We also have some interesting plots from our data under [docs/analysis.md](https://gitlab.com/bahnvorhersage/bahnvorhersage/-/blob/master/docs/analysis.md).  
You can also generate plots on our [website](https://bahnvorhersage.de/stats/stations#content).

## Running webserver

If you for whatever reason want to run our website on your computer just do as described below.  
But you are going to need a connection to our database, to do so [contact us...](mailto:info@bahnvorhersage.de)

To run our webserver we strongly recommend to use Docker.

First, the usergroup 420 has to have the rights to write to the cache volume. In order to add the permission, do the following
```bash
sudo chown -R :420 /path/to/your/cache/
```

Then in the project directory run:

```bash
# In order to build:
DOCKER_BUILDKIT=1 docker build -f webserver/Dockerfile.webserver . -t webserver
# In order to serve:
docker run -p 5000:5000 -v $(pwd)/config.py:/mnt/config/config.py -v $(pwd)/cache:/usr/src/app/cache webserver
```
The webserver should now be running on http://localhost:5000/

## Installing Cartopy

We use cartopy in our backend to generate nice looking geo plots. It can be hard to install cartopy.
Modern Cartopy uses proj 8, which is unavailable in many package repositories. Here is how to install in from source:

Install build dependencies
```bash
apt update -y
apt install -y --fix-missing --no-install-recommends \
            software-properties-common build-essential ca-certificates \
            make cmake wget unzip libtool automake \
            zlib1g-dev libsqlite3-dev pkg-config sqlite3 libcurl4-gnutls-dev \
            libtiff5-dev git
```

Clone proj repository from GitHub
```bash
git clone https://github.com/OSGeo/PROJ.git
cd PROJ
```

Build and install
```bash
./autogen.sh
./configure
make
make install
```

After installing Proj, you should now be able to just install cartopy
```bash
pip install cartopy
```

## Credits

- Marius De Kuthy Meurers aka [NotSomeBot](https://github.com/mariusdkm)
- Theo Döllman aka [McToel](https://gitlab.com/mctoel)
  
