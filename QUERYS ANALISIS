
-- 1. Personas con al menos una compra
SELECT DISTINCT p.nombre, p.app, p.apm 
FROM persona p 
JOIN compra c ON p.idpersona = c.idpersona;

-- 2. Mujeres nacidas en 2003 con compras
SELECT DISTINCT p.nombre, p.app, p.apm 
FROM persona p 
JOIN compra c ON p.idpersona = c.idpersona 
WHERE p.anoNacimiento = 2003 AND p.genero = 'F';

-- 3. Hombres nacidos antes de 2000
SELECT DISTINCT p.nombre, p.app, p.apm 
FROM persona p 
WHERE p.anoNacimiento < 2000 AND p.genero = 'M';

-- 4. Personas nacidas entre 1980 y 1990
SELECT p.nombre, p.app, p.apm 
FROM persona p 
WHERE p.anoNacimiento BETWEEN 1980 AND 1990 
ORDER BY p.app, p.apm, p.nombre;

-- 5. Compras mayores a 15000
SELECT * 
FROM compra 
WHERE totalcompra > 15000;

-- 6. Compras entre enero y abril 2022
SELECT DISTINCT p.nombre, p.app, p.apm 
FROM persona p 
JOIN compra c ON p.idpersona = c.idpersona 
WHERE c.fechacompra BETWEEN '2022-01-01' AND '2022-04-01';

-- 7. Nombre inicia con A, nacidos en 1988
SELECT DISTINCT p.nombre, p.app, p.apm 
FROM persona p 
JOIN compra c ON p.idpersona = c.idpersona 
WHERE p.nombre LIKE 'A%' AND p.anoNacimiento = 1988;

-- 8. Tickets de compras hechas por hombres
SELECT c.numeroticket 
FROM compra c 
JOIN persona p ON c.idpersona = p.idpersona 
WHERE p.genero = 'M';

-- 9. Compras <2000 o personas nacidas antes de 1990
SELECT * 
FROM compra c 
JOIN persona p ON c.idpersona = p.idpersona 
WHERE c.totalcompra < 2000 OR p.anoNacimiento < 1990;

-- 10. Compras antes de 2022 ordenadas
SELECT * 
FROM compra 
WHERE fechacompra < '2022-01-01' 
ORDER BY totalcompra DESC;

-- 11. IDs de personas con compras
SELECT DISTINCT p.idpersona 
FROM persona p 
JOIN compra c ON p.idpersona = c.idpersona;

-- 12. Compras >5000 con ticket que termina en 3
SELECT * 
FROM compra 
WHERE totalcompra > 5000 AND numeroticket LIKE '%3';

-- 13. Personas sin compras
SELECT p.nombre, p.app, p.apm
FROM persona p 
LEFT JOIN compra c ON p.idpersona = c.idpersona
WHERE c.idcompra IS NULL;
