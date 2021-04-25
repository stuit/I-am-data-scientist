-- 4.1
-- SELECT city city_name,
--       count(*) cnt
-- FROM dst_project.airports
-- GROUP BY city_name
-- ORDER BY 2 DESC;

-- 4.2.1
-- SELECT count(DISTINCT status)
-- FROM dst_project.flights;

-- 4.2.2
-- SELECT count(*)
-- FROM dst_project.flights
-- WHERE status = 'Departed';

-- 4.2.3
-- SELECT count(*)
-- FROM dst_project.seats s
-- JOIN dst_project.aircrafts a ON s.aircraft_code = a.aircraft_code
-- WHERE s.aircraft_code = '773';

-- 4.2.4
-- SELECT count(DISTINCT flight_id)
-- FROM dst_project.flights
-- WHERE status = 'Arrived'
--   AND date(actual_arrival) BETWEEN '2017-04-01' AND '2017-09-01';

-- 4.3.1
-- SELECT count(DISTINCT flight_id)
-- FROM dst_project.flights
-- WHERE status = 'Cancelled';

-- 4.3.2.a
-- SELECT model aircraft_model
-- FROM dst_project.aircrafts
-- WHERE (model LIKE '%Boeing%'
--      OR model LIKE '%Sukhoi Superjet%'
--      OR model LIKE '%Airbus%')
-- GROUP BY model;

-- 4.3.2.b
-- SELECT (CASE
--             WHEN a.model LIKE 'Boeing%' THEN 'Boeing'
--             WHEN a.model LIKE 'Airbus%' THEN 'Airbus'
--             WHEN a.model LIKE 'Sukhoi Superjet%' THEN 'Sukhoi Superjet'
--         END) aircrafts,
--       count(*) aircraft_count
-- FROM dst_project.aircrafts a
-- GROUP BY (CASE
--               WHEN a.model LIKE 'Boeing%' THEN 'Boeing'
--               WHEN a.model LIKE 'Airbus%' THEN 'Airbus'
--               WHEN a.model LIKE 'Sukhoi Superjet%' THEN 'Sukhoi Superjet'
--           END);

-- 4.3.3
-- SELECT (CASE
--             WHEN a.timezone LIKE 'Asia%' THEN 'Asia'
--             WHEN a.timezone LIKE 'Europe%' THEN 'Europe'
--             WHEN a.timezone LIKE 'Australia%' THEN 'Australia'
--         END) AS timezones,
--       count(*) airport_count
-- FROM dst_project.airports a
-- GROUP BY (CASE
--               WHEN a.timezone LIKE 'Asia%' THEN 'Asia'
--               WHEN a.timezone LIKE 'Europe%' THEN 'Europe'
--               WHEN a.timezone LIKE 'Australia%' THEN 'Australia'
--           END);

-- 4.3.4
-- SELECT DATE_PART('hour', actual_arrival - scheduled_arrival),
--       flight_id
-- FROM dst_project.flights
-- WHERE status = 'Arrived'
-- ORDER BY 1 DESC;

-- 4.4.1
-- SELECT min(scheduled_departure)
-- FROM dst_project.flights;

-- 4.4.2
-- SELECT max(EXTRACT(epoch
--                   FROM (scheduled_arrival - scheduled_departure))) / 60
-- FROM dst_project.flights;

-- 4.4.3
-- SELECT (scheduled_arrival - scheduled_departure) flight_time,
--       departure_airport,
--       arrival_airport
-- FROM dst_project.flights
-- ORDER BY 1 DESC;

-- 4.4.4
-- SELECT avg(EXTRACT(epoch
--                   FROM (scheduled_arrival - scheduled_departure))) / 60
-- FROM dst_project.flights;

-- 4.5.1
-- SELECT count(*) seat_count,
--       fare_conditions
-- FROM dst_project.seats
-- WHERE aircraft_code = 'SU9'
-- GROUP BY fare_conditions;

-- 4.5.2
-- SELECT min(total_amount) 
-- FROM dst_project.bookings;

-- 4.5.3
-- SELECT b.seat_no
-- FROM dst_project.tickets t
-- JOIN dst_project.boarding_passes b ON t.ticket_no = b.ticket_no
-- WHERE t.passenger_id = '4313 788533';

-- 5.1.1
-- SELECT count(*)
-- FROM dst_project.flights
-- WHERE arrival_airport = 'AAQ'
--   AND status = 'Arrived'
--   AND actual_arrival >= '2017-01-01';

-- 5.1.2
-- SELECT count(*)
-- FROM dst_project.flights
-- WHERE departure_airport = 'AAQ'
--   AND (date_trunc('month', scheduled_departure) in ('2017-01-01','2017-02-01', '2017-12-01'))
--   AND status = 'Arrived';

-- 5.1.3
-- SELECT count(*)
-- FROM dst_project.flights
-- WHERE departure_airport = 'AAQ'
--   AND status = 'Cancelled';

-- 5.1.4
-- SELECT count(flight_no)
-- FROM dst_project.flights
-- WHERE departure_airport = 'AAQ'
--   AND arrival_airport not in ('SVO', 'VKO', 'DME');

-- 5.1.5
-- WITH planes AS
--   (SELECT f.aircraft_code,
--           a.model
--   FROM dst_project.flights f
--   JOIN dst_project.aircrafts a ON f.aircraft_code = a.aircraft_code
--   WHERE f.departure_airport = 'AAQ'
--   GROUP BY f.aircraft_code,
--             a.model)
-- SELECT count(*) seat_num,
--       p.model
-- FROM planes p
-- JOIN dst_project.seats s ON p.aircraft_code = s.aircraft_code
-- GROUP BY p.model;

------------
-- PROJECT-4
WITH anapa AS -- Данные по направлению Анапа за зимний период 2017г.
  (SELECT *
   FROM dst_project.flights
   WHERE departure_airport = 'AAQ' -- Делаем выборку только по г.Анапа
     AND (date_trunc('month', scheduled_departure) in ('2017-01-01', '2017-02-01', '2017-12-01')) -- Выборка на зимний период
     AND status not in ('Cancelled') ), -- По актуальным рейсам
     
     plane_seats AS -- Общее количество мест в самолете
  (SELECT count(*) seats_num,
          aircraft_code
   FROM dst_project.seats
   GROUP BY aircraft_code),
   
     plane_fares AS -- Данные о количестве мест по классам
  (WITH fares AS
     (SELECT count(*) seat_num,
             aircraft_code,
             fare_conditions
      FROM dst_project.seats
      GROUP BY fare_conditions,
               aircraft_code
      ORDER BY aircraft_code) 
   SELECT f1.aircraft_code,
          f1.seat_num economy, -- количество мест в экономе
          coalesce(f2.seat_num, 0) business, -- количество мест в бизнесе
          coalesce(f3.seat_num, 0) comfort -- количество мест в первом классе
   FROM fares f1
   LEFT JOIN fares f2 ON f1.aircraft_code = f2.aircraft_code
   AND f2.fare_conditions = 'Business'
   LEFT JOIN fares f3 ON f1.aircraft_code = f3.aircraft_code
   AND f3.fare_conditions = 'Comfort'
   WHERE f1.fare_conditions = 'Economy' ),
   
     income AS -- Данные о продаже билетов 
  (SELECT count(*) sold_tickets, -- количество проданных билетов
          sum(amount) total_income, -- общяя выручка от продажи
          flight_id
   FROM dst_project.ticket_flights
   GROUP BY flight_id),
   
     airport_dist AS -- Расстояние между аэропортами
  (SELECT f.flight_id,
          (ACOS(SIN(a.latitude*PI()/180)*SIN(a1.latitude*PI()/180) + COS(a.latitude*PI()/180)*COS(a1.latitude*PI()/180)*COS(a1.longitude*PI()/180-a.longitude*PI()/180)) * 6371.000) distance
   FROM dst_project.flights f
   LEFT JOIN dst_project.airports a ON f.departure_airport = a.airport_code
   LEFT JOIN dst_project.airports a1 ON f.arrival_airport = a1.airport_code),
   
     fuel_cons AS -- Данные о расходе топлива за час полета
  (SELECT (CASE
               WHEN a.aircraft_code LIKE '733' THEN 3009 -- средний расход топлива за час Boeing 737-300
               WHEN a.aircraft_code LIKE 'SU9' THEN 2426 -- средний расход топлива за час Sukhoi Superjet-100
           END) AS litres_per_hour,
          a.aircraft_code
   FROM dst_project.aircrafts a
   WHERE a.aircraft_code = '733'
     OR a.aircraft_code = 'SU9' ) -- выборка идет только по бортам совершающие по направлению г.Анапа
SELECT row_number()over (order by an.flight_id),
       *,
       (an.scheduled_arrival - an.scheduled_departure) flight_time,
       (EXTRACT(epoch
                FROM (an.scheduled_arrival - an.scheduled_departure)) / 3600)*c.litres_per_hour fuel_consumption,
       ((EXTRACT(epoch
                 FROM (an.scheduled_arrival - an.scheduled_departure)) / 3600)*c.litres_per_hour)/1000*60758 approx_fuel_cost -- 60758.00 руб расход за тонну топлива и обслуживание по заправке  
FROM anapa an
LEFT JOIN dst_project.aircrafts a USING(aircraft_code)
LEFT JOIN plane_seats s USING(aircraft_code)
LEFT JOIN plane_fares f USING(aircraft_code)
LEFT JOIN income i USING(flight_id)
LEFT JOIN airport_dist ad USING(flight_id)
LEFT JOIN fuel_cons c USING(aircraft_code)
ORDER BY an.flight_id













