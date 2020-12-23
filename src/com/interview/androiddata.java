FILES-
Files are a quick way to persist data in Android.
• The internal storage is a great place to store files that are specific and private to
your app.
• Use the internal cache to store temporary files.
• Use external storage to store files that you want users or other apps to have access
to.
• External files are persistent, even after the app is uninstalled.
• Internal files are deleted when the app is uninstalled.
• To write files to the external storage, the correct permissions must be set in the
manifest.
• Because the external storage is not secure, it’s a good idea to use AES encryption
with a password-based generated key.
• As a general security practice, apps that store data files should not be allowed to
be installed on external storage.
• Serializable and Parcelable are ways of marshaling data for transport or storage.
• Do not use Parcelable to store persistent data; if the Parcel is changed, the old
data will not be read properly.

SHARED PREFERENCES-
SharedPreferences are a great way to store small bits of data for an app.
• User—specific app configuration data is often stored in SharedPreferences.
• Preferences are also stored in an app's default SharedPreferences, but they are
not to be confused with each other.
• SharedPreferences are not a good choice to store structured data, like various
entities, use a database for that.
• SharedPreferences are organized in key—value pairs that are ultimately written
out to a file in .xml format.
• Each app has its own SharedPreferences. When the app is uninstalled or the data
is deleted, these stored values are wiped away.
• You can create custom-named SharedPreferences files and store user's choices
in them.

SQLITE DATABASE-
 When creating an SQLite database, you should define a database schema — a
group of String constants, used to define your database structure.
• A database needs to be created and updated, according to the context of the app,
and the version of the database.
• If an app doesn't have a database with the same name as in your schema, it will
create one, using the defined schema.
• If the app already has a database with the same name, it will run the update
process, but only if the database version changed.
• You should avoid dropping the database if it changes, and try to migrate the
structure between versions.
• Every database consists of the four standard operations - Create, Read, Update,
and Delete, or CRUD for short.
• To help you avoid so much SQL code, and to simplify the operations, Android
utilizes the SQLiteOpenHelper.
To store data for operations, the SQLiteOpenHelper uses ContentValues.
• ContentValues are just a key-value pair structure, just like a Map, which is used to
insert or update data in the database.
• You can inspect the database by copying it from the Device File Explorer, and
opening it with a tool, like the SQLite Browser or DB Browser.
• It is a good practice to write some Unit tests for your database, to be sure
everything works as expected, without running the application.

CONTENT PROVIDER-
Content providers sit just above the data source of the app, providing an
additional level of abstraction from the data repository.
• A content provider allows for the sharing of data between apps
• A content provider can be a useful way of making a single data provider if the data
is housed in multiple repositories for the app.
• It is not necessary to use a content provider if the app is not intended to share
data with other apps.
• The content resolver is utilized to run queries on the content provider.
• Using a content provider can allow granular permissions to be set on the data.
• Use best practices such as selection clauses and selection arguments to prevent
SQL Injection when utilizing a content provider, or any raw query data
mechanism.
• Data in a content provider can be accessed via exposed URIs that are matched by
the content provider.

Room vs Anko-
Anko SQLite provides a nice API to manage your data persistence layer but you still have
to do the heavy lifting by yourself. While Room is more like a framework. Generates
databases from annotated classes for you provides observable queries and has really nice
testing support. Also works well with Android Architecture Components.
Realm-
The difference in getting related data between Room and Realm is enormous. Getting
happens the same way as with a simple access to the property of the object. An
important point – when implementing a many-to-many communication, the connection
between objects is unidirectional by default. For two-way dependency between objects,
you must use the @LinkingObjects annotation.
object-oriented database, the conversion of
application models to save them in the database requires minimal effort from the
developer. First of all, the model class must be accessible for inheritance (in Java,
nothing needs to be changed, in Kotlin – add the open modifier). Also, this class should
be the descendant of the abstract class RealmObject, which encapsulates the logic of
interaction with the database.

Room Database-
Contains the database holder and servers as the main access point for the underlying
connection to your app’s persisted relational data.
To create a database we need to define an abstract class that extends RoomDatabase.
This class is annotated with @Database, lists the entities contained in the database, and
the DAO’s which access them.
The class that’s annotated with @Database should satisfy the following conditions:
Be an abstract class that extends RoomDatabase .
Include the list of entities associated with the database within the annotation.
Contain an abstract method that has 0 arguments and returns the class that is
annotated with @Dao .
At runtime, you can acquire an instance of Database by calling
Room.databaseBuilder() or Room.inMemoryDatabaseBuilder() .
Entity-
Represents a table within the database. Room creates a table for each class that has an
@Entity annotation, the fields in the class correspond to columns in the table.
Therefore, the entity classes tend to be small model classes that don’t contain any logic.
@Entity — every model class with this annotation will have a mapping table in DB
foreignKeys — names of foreign keys
indices — list of indicates on the table
primaryKeys — names of entity primary keys
tableName
@PrimaryKey — as its name indicates, this annotation points the primary key of the
entity. autoGenerate — if set to true, then SQLite will be generating a unique id for the
column
@PrimaryKey(autoGenerate = true)
@ColumnInfo — allows specifying custom information about column
@ColumnInfo(name = “column_name”)
@Ignore — field will not be persisted by Room
@Embeded — nested fields can be referenced directly in the SQL queries.
Room Dao-
DAOs are responsible for defining the methods that access the database. In the initial
SQLite, we use the Cursor objects. With Room, we don’t need all the Cursor related
code and can simply define our queries using annotations in the Dao class.
Open build.gradle file (project level) and add the following line:
Open the build.gradle file (module:app) and add these lines:
Converters: because Bitmap is used in Entity, and Bitmap as such is a complex
object, and complex objects are not supported in databases, so it needs to be converted
to ByteArray, and Room needs to know about this converter by @TypeConverter
annotation as well as acknowledging it in @Database class.

ROOM LIBRARY-
Compile-time verification of SQL queries, @Query, and @Entity objects, which
prevents runtime app crashing
Easily integrated with other Architecture components (like LiveData), in other
words, it is built to work with LiveData and RxJava for data observation while
SQLite is not
Room is an ORM developed by Google as a part of Jetpack's architecture
components to simplify the interaction with your SQLite database and to reduce
the amount of boilerplate code.
• Entities in Room represent tables in your database.
• DAO stands for Data Access Object.
• The Repository class handles the interaction with your Room database and other
backend endpoints.
• The ViewModel communicates the data coming from your repository to your views
and has the advantage of surviving configuration changes since it's lifecycleaware.
• LiveData is a data holder class that can hold information and be observed for
changes.
• ORM stands for Object Relational Mapper.
• Shared preferences are very useful when you need to store and share simple data
such as user preferences as key-value pairs.
• The main disadvantage of using shared preferences is that you can't store large
amounts of data since it's not efficient and there's no way to use queries to search
for information.
SQLite is a fast and lightweight local database natively supported by Android that
allows you to store large amounts of data in a structured way.
• SQLite is available for most platforms and can be used with many popular
programming languages.
• Because SQLite is lightweight, it's suitable for devices with restricted memory
such as smartphones and smart TVs.
• ORMs provide an additional layer of abstraction that allows you to interact with
your relational database with an Object-Oriented Language syntax.

ENTITY-
Tables are structures similar to spreadsheets or two-dimensional arrays that let
you store records objects, as rows with one or more fields defined as columns.
• The @Entity annotation declares that the annotated class is a Room entity, and
you will need to generate a table.
• The @PrimaryKey annotation allows you to define a primary key for your table to
uniquely differentiate data.
The @ColumnInfo annotation lets you change the names for your columns, so you
can use different naming conventions in Kotlin and in SQL.
• The @Ignore annotation tells Room to ignore a certain property from your class so
it does not get converted into a column in the database.
• The @Embedded annotation can be used on an entity's field to tell Room that the
properties on the annotated object should be represented as columns on the same
entity. This way you can organize your data clearly while writing the same SQL.

RELATIONS-
An entity relation diagram, ER diagram or ERD is a kind of flowchart that
illustrates the relation between the components of a system.
• Entities represent a component, object or a concept of a system. They are usually
translated as tables in your database.
• Entities in Crow's Foot notation also include a list of attributes or properties that
define them.
• An attribute that can uniquely identify a record of your entity is known as a key
attribute and they usually become primary keys in your database.
• A relationship tells you how two entities interact with each other and it is usually
represented as a verb.
• The cardinality of an ERD tells you the kind of relationship that two entities have.
• One to one relationship: When one entity can only be related to one and only
one instance of the other entity.
• One to many relationship: When one entity can be related to many instances of
another entity.
• Many to many relationship: When many instances of an entity can also be
related to many instance of another entity.

DAO PATTERN-
Database Access Objects are commonly referred to as DAOs.
• DAOs are objects that provide access to your app's data by abstracting most of the
complexity behind querying and updating your database.
• In Room, DAOs can be defined as interfaces or abstract classes.
• The @Insert annotation is a marker, which allows you to automatically create an
INSERT INTO query.
• @Insert can take an OnConflictStrategy parameter, that allows you to specify
what happens in case there is a conflict when creating a new database entry.
• @Query allows you to perform any kind of queries in your database. You can also
use autocomplete, to easily connect to entities and their property definitions.
• @Transaction tells Room that the following SQL statements should be executed
in a single transaction.
• @Delete allows you to automatically create DELETE FROM queries, but it requires
a parameter to be removed; e.g., a Question object.
• @Update updates a record in the database if it already exists, or omits the
changes, if it doesn't, leaving the database unchanged.
• Writing tests with Espresso is a good way to see if your database code works
properly.
• You can run Espresso tests, without manually going through the app, in less
than a few seconds.
• Inserting the data and reading from the databse in Espresso is safe, because you
can work with an in-memory version of the database.
• In-memory databases clear up after tests end, so there's no need to do extra
cleanup, other than to close() the database, to avoid leaks.

ROOM WITH ARCHITECTURE COMPONENTS-VIEWMODEL & LIVEDATA 
LiveData is a data holder class, like a List, that can be observed for changes by an
Observer.
• LiveData is lifecycle-aware, meaning it can observe the lifecycle of Android
components like the Activity or Fragment. It will only keep updating observers if
its component is still active.
• The ViewModel is part of the Google's architecture components and it's
specifically designed to manage data related to your user interface.
• A Repository helps you separate concerns to have a single entry point for your
app's data.
• You can combine LiveDatas and add different sources, to take action if
something changes.

ROOM MIGRATION-
Simply put, a database migration or schema migration is the process of moving
your data from one database schema to another.
• SQLite handles database migrations by specifying a version number for each
database schema that you create.
• Room provides an abstraction layer on top of the traditional SQLite migration
methods with Migration.
• Migration(startVersion, endVersion) is the base class for a database
migration. It can move between any two migrations defined by the startVersion
and endVersion parameters.
• fallbackToDestructiveMigration() tells Room to destructively recreate tables
if you haven't specified a migration.

FIREBASE-
Firebase was a mobile backend service that let you build mobile apps
without having to worry about managing your own backend.
Data between different Firebase products is shared where and when needed,
which leads to even faster development.
Authentication — user login and identity
Realtime Database — realtime, cloud hosted, NoSQL database
Cloud Firestore — realtime, cloud hosted, NoSQL database
Cloud Storage — massively scalable file storage
Cloud Functions — “serverless”, event driven backend
Firebase Hosting — global web hosting
ML Kit —SDK for common ML tasks
Analytics — understand your users, and how they use your app
Predictions — apply machine learning to analytics to predict user behavior
Cloud Messaging — send messages and notifications to users
Remote Config — customize your app without deploying a new version; monitor the
changes
A/B Testing — run marketing and usability experiments to see what works best
Dynamic Links — enable native app conversions, user sharing, and marketing
campaigns
App Indexing — re-engage users with Google Search integration
In-App Messaging — engage your active users with targeted messages
Test Lab — scalable and automated app testing on cloud-hosted devices
Crashlytics — get clear, actionable insight into your app’s crashes
Performance Monitoring — gain insight into your app’s performance issues

FIREBASE REALTIME DATABASE-
The Realtime Database provides database rules to control access for each user.
• Firebase Authentication is very much connected to database solutions Firebase
offers, since it controls the access to data, on a per-user-basis.
• Firebase Realtime Database stores data in JSON format.
• Because the structure is a JSON, it operates with simple data like numbers, strings,
objects, booleans and lists.
• Firebase Realtime Database data structure should be as flat as possible and
designed with scaling in mind.
FirebaseDatabase object is the main entry point to the database
• DatabaseReference class represents a particular location in the database and it is
used to refer to the location in the database to which you want to write to or read
from.
• push() method is used to create an empty node with an auto-generated key.
• Firebase Realtime Database has several types of listeners, and each listener type
has a different kind of callback.
• ValueEventListener listens for data changes to a specific database reference.
• ChildEventListener listens for changes to the children of a specific database
reference.
• You need to decide how to handle listeners when the user is not actively
interacting with the app. In most cases, you want to stop listening for updates. To
do that you need to remove the listener.
• For updating data in Realtime Database, the setValue() method is used.
• You can delete data by using the setValue method and specify null as an
argument or you can use removeValue() method which will set the value at the
specified location to null.
• A query is a request for data or information from a database. Query class is used
for reading data and it has many useful methods that allow you to fetch the data in
a way you want.
• A database transaction is a unit of work that is independently executed and it
must be atomic, consistent, isolated and durable.
• To improve query performance you should consider defining indexing rules.

Realtime Offline-
ealtime database allows you to enable disk persistence to make your data
available when you're offline.
• Enabling disk persistence also tracks of all the writes you initiated while you
were offline and then when network connection comes back it synchronizes all
the write operations.
• Realtime database stores a copy of the data, locally, only for active listeners. You
can use the keepSynced method on a database reference to save data locally for
the location that has no active listeners attached.
• Firebase provides you with the real-time presence system, which allows your app
to know the status of your users, are they online or offline.
• Firebase handles latency in a way that stores a timestamp, that is generated on
the server, as data when the client disconnects and lets you use that data to
reliably know the exact time when the user disconnected.

CLOUD FIRESTORE-
Cloud Firestore is a NoSQL database similar to the Realtime Database.
• Firestore stores data as a collection of objects which are stored in a hierarchical
structure that resemble a tree.
• Documents and collections are the main building blocks of Cloud Firestore.
• Documents consist of key-value pairs known as fields.
• Collections are a group of documents.
• Collections can only contain documents.
• The root of the Cloud Firestore database can only consist of collections.
• A document cannot contain other documents, but it can contain another
collection; these are known as subcollections.
It’s easier to query, filter and sort data using the Firestore since it can all be done
within a single request.
• It’s best to use foreign-key-like fields in objects, as you don’t want to duplicate
data and clutter the database.
• The Firestore scales horizontally; this is easier than the Realtime Database which
scales vertically.

Managing Data With Cloud Firestore-
 Firestore database is created in the Firebase console.
• You need to add a Firestore client library for Android to the project in order to use
Firestore APIs.
• You need to initialize a Firestore instance in order to communicate with the
database.
• You call the collection method passing in the collection path to get a reference
to the collection at the specified path in the database.
• You need to create and populate the map of data that you wan't to save to the
database.
• You call the set method on the document reference that will replace the data in
the document if it already exists or it will create it if it doesn't to save the data to
the database. You pass in the map that contains the data that you want to write to
that document.
• Firebase supports transactions which are used in cases when you want to write a
bunch of data to the database at once.
• You call the update method on a document reference to update fields in the
document.
• You call the delete method on a document reference which deletes the document
referred to by the reference.
• Adding, updating and deleting operations are asynchronous.
• You can use the Firebase console to manage data in the Firestore database.

Reading from firestore-
• Firestore allows to read data once or to listen for data changes in real-time.
• To get the data once, you would need to use get method on the collection
reference.
• ListenerRegistration interface represents a Firestore subscription listener.
• You can call addSnapshotListener() on a collection reference to start listening
for data changes at a specific location.
• Queries are used to get only a subset of the documents within a collection.
• Cloud Firestore stores a copy of data that your app is using, locally, so that you can
access the data, if the device goes offline.
• You can also use orderBy() and limit(), on the collection reference, to get only
specific documents from a collection.
• Pagination allows you to split your database data into chunks so that you don't
need to fetch all your data at once.
• To ensure good performance for every query, Firestore requires an index, when
creating them.

Securing data in firestore-
Security rules check the requests that are coming to the database and lets
through those that satisfy the criteria and reject the ones that don't.
• Security rules consist of two things: 1. Specifying which documents you are
securing; 2. What logic you're using to secure them.
• In the Rules tab in the Firebase console, you can see your current security
configuration.
• match statement specifies the path to the document.
• allow expression specifies when the writing or reading the data is allowed.
• Security rules in Cloud Firestore do not cascade.
• Cloud Firestore provides Simulator feature that you can use to test your rules.

Cloud storage-
• Cloud Storage is a Firebase product used for saving files associated with your app.
• If you lose a network connection in the middle of the upload or a download, the
transfer will continue where it left off after you reconnect to the network.
• Cloud Storage also has security features that will make your files secure.
• The foundations of the Cloud Storage are folders that you can create to organize
your data.





