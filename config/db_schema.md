# Esquema de la Base de Datos de Ventas

## Tabla: clientes
- id: INTEGER (PRIMARY KEY)
- nombre: VARCHAR(100) - Nombre completo del cliente
- email: VARCHAR(100) - Correo electrónico
- pais: VARCHAR(50) - País de origen del cliente
- fecha_registro: DATE - Fecha en la que se registró el cliente

## Tabla: productos
- id: INTEGER (PRIMARY KEY)
- nombre: VARCHAR(100) - Nombre del artículo
- categoria: VARCHAR(50) - Categoría (Muebles, Tecnología)
- precio: DECIMAL(10,2) - Precio unitario del producto
- stock: INTEGER - Unidades en inventario

## Tabla: ventas
- id: INTEGER (PRIMARY KEY)
- cliente_id: INTEGER - Relaciona con clientes.id
- producto_id: INTEGER - Relaciona con productos.id
- cantidad: INTEGER - Cantidad de unidades vendidas
- total: DECIMAL(10,2) - Importe total de la venta (precio * cantidad)
- fecha_venta: DATE - Fecha de la transacción
