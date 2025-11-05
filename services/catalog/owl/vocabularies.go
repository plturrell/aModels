package owl

// Standard RDF/OWL Vocabulary URIs
// These are commonly used vocabularies for semantic web metadata

const (
	// RDF namespace
	RDFNamespace = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

	// RDFS namespace
	RFSNamespace = "http://www.w3.org/2000/01/rdf-schema#"

	// OWL namespace
	OWLNamespace = "http://www.w3.org/2002/07/owl#"

	// XSD namespace for data types
	XSDNamespace = "http://www.w3.org/2001/XMLSchema#"

	// Dublin Core namespace
	DCNamespace = "http://purl.org/dc/elements/1.1/"

	// Dublin Core Terms namespace
	DCTermsNamespace = "http://purl.org/dc/terms/"

	// SKOS namespace
	SKOSNamespace = "http://www.w3.org/2004/02/skos/core#"

	// ISO 11179 namespace (base, will be customized per registry)
	ISO11179Namespace = "http://amodels.org/iso11179#"
)

// Standard RDF/RDFS/OWL Classes
const (
	RDFType           = RDFNamespace + "type"
	RDFProperty       = RDFNamespace + "Property"
	RDFStatement      = RDFNamespace + "Statement"
	RDFSubject        = RDFNamespace + "subject"
	RDFPredicate      = RDFNamespace + "predicate"
	RDFObject         = RDFNamespace + "object"
	RDFList           = RDFNamespace + "List"
	RDFNil            = RDFNamespace + "nil"
	RDFFirst          = RDFNamespace + "first"
	RDFRest           = RDFNamespace + "rest"
	RDFValue          = RDFNamespace + "value"
	RDFAlt            = RDFNamespace + "Alt"
	RDFBag            = RDFNamespace + "Bag"
	RDFSeq            = RDFNamespace + "Seq"

	RDFSLabel         = RFSNamespace + "label"
	RDFSComment       = RFSNamespace + "comment"
	RDFSSeeAlso       = RFSNamespace + "seeAlso"
	RDFSIsDefinedBy   = RFSNamespace + "isDefinedBy"
	RDFSDomain        = RFSNamespace + "domain"
	RDFSRange         = RFSNamespace + "range"
	RDFSSubClassOf    = RFSNamespace + "subClassOf"
	RDFSSubPropertyOf = RFSNamespace + "subPropertyOf"
	RDFSMember        = RFSNamespace + "member"
	RDFSLiteral       = RFSNamespace + "Literal"
	RDFSResource      = RFSNamespace + "Resource"
	RDFSClass         = RFSNamespace + "Class"
	RDFSContainer     = RFSNamespace + "Container"
	RDFSContainerMembershipProperty = RFSNamespace + "ContainerMembershipProperty"
	RDFSDatatype      = RFSNamespace + "Datatype"

	OWLClass                  = OWLNamespace + "Class"
	OWLObjectProperty         = OWLNamespace + "ObjectProperty"
	OWLDatatypeProperty      = OWLNamespace + "DatatypeProperty"
	OWLFunctionalProperty    = OWLNamespace + "FunctionalProperty"
	OWLInverseFunctionalProperty = OWLNamespace + "InverseFunctionalProperty"
	OWLTransitiveProperty    = OWLNamespace + "TransitiveProperty"
	OWLSymmetricProperty     = OWLNamespace + "SymmetricProperty"
	OWLAsymmetricProperty    = OWLNamespace + "AsymmetricProperty"
	OWLReflexiveProperty     = OWLNamespace + "ReflexiveProperty"
	OWLIrreflexiveProperty   = OWLNamespace + "IrreflexiveProperty"
	OWLThing                 = OWLNamespace + "Thing"
	OWLNothing              = OWLNamespace + "Nothing"
	OWLAllValuesFrom         = OWLNamespace + "allValuesFrom"
	OWLSomeValuesFrom        = OWLNamespace + "someValuesFrom"
	OWLHasValue              = OWLNamespace + "hasValue"
	OWLMinCardinality        = OWLNamespace + "minCardinality"
	OWLMaxCardinality        = OWLNamespace + "maxCardinality"
	OWLCardinality           = OWLNamespace + "cardinality"
	OWLEquivalentClass       = OWLNamespace + "equivalentClass"
	OWLDisjointWith          = OWLNamespace + "disjointWith"
	OWLIntersectionOf        = OWLNamespace + "intersectionOf"
	OWLUnionOf               = OWLNamespace + "unionOf"
	OWLComplementOf          = OWLNamespace + "complementOf"
	OWLOneOf                  = OWLNamespace + "oneOf"
	OWLOnProperty            = OWLNamespace + "onProperty"
	OWLRestriction            = OWLNamespace + "Restriction"
	OWLEquivalentProperty    = OWLNamespace + "equivalentProperty"
	OWLInverseOf             = OWLNamespace + "inverseOf"
	OWLSameAs                = OWLNamespace + "sameAs"
	OWLDifferentFrom         = OWLNamespace + "differentFrom"
	OWLAllDifferent          = OWLNamespace + "AllDifferent"
	OWLDistinctMembers       = OWLNamespace + "distinctMembers"
	OWLVersionInfo           = OWLNamespace + "versionInfo"
	OWLBackwardCompatibleWith = OWLNamespace + "backwardCompatibleWith"
	OWLIncompatibleWith      = OWLNamespace + "incompatibleWith"
	OWLDeprecated            = OWLNamespace + "deprecated"
	OWLPriorVersion          = OWLNamespace + "priorVersion"
	OWLImports               = OWLNamespace + "imports"
	OWLOntology              = OWLNamespace + "Ontology"
)

// Standard XSD Data Types
const (
	XSDString      = XSDNamespace + "string"
	XSDInteger     = XSDNamespace + "integer"
	XSDDecimal     = XSDNamespace + "decimal"
	XSDDouble      = XSDNamespace + "double"
	XSDFloat       = XSDNamespace + "float"
	XSDBoolean     = XSDNamespace + "boolean"
	XSDDate        = XSDNamespace + "date"
	XSDTime        = XSDNamespace + "time"
	XSDDateTime    = XSDNamespace + "dateTime"
	XSDDuration    = XSDNamespace + "duration"
	XSDAnyURI      = XSDNamespace + "anyURI"
	XSDBase64Binary = XSDNamespace + "base64Binary"
	XSDHexBinary   = XSDNamespace + "hexBinary"
	XSDLong        = XSDNamespace + "long"
	XSDInt         = XSDNamespace + "int"
	XSDShort       = XSDNamespace + "short"
	XSDByte        = XSDNamespace + "byte"
	XSDUnsignedLong = XSDNamespace + "unsignedLong"
	XSDUnsignedInt = XSDNamespace + "unsignedInt"
	XSDUnsignedShort = XSDNamespace + "unsignedShort"
	XSDUnsignedByte = XSDNamespace + "unsignedByte"
	XSDNonNegativeInteger = XSDNamespace + "nonNegativeInteger"
	XSDNonPositiveInteger = XSDNamespace + "nonPositiveInteger"
	XSDNegativeInteger = XSDNamespace + "negativeInteger"
	XSDPositiveInteger = XSDNamespace + "positiveInteger"
)

// Dublin Core Properties
const (
	DCTitle       = DCNamespace + "title"
	DCDescription = DCNamespace + "description"
	DCCreator     = DCNamespace + "creator"
	DCPublisher   = DCNamespace + "publisher"
	DCContributor = DCNamespace + "contributor"
	DCDate        = DCNamespace + "date"
	DCType        = DCNamespace + "type"
	DCFormat      = DCNamespace + "format"
	DCIdentifier  = DCNamespace + "identifier"
	DCSource      = DCNamespace + "source"
	DCLanguage    = DCNamespace + "language"
	DCRelation    = DCNamespace + "relation"
	DCCoverage    = DCNamespace + "coverage"
	DCRights      = DCNamespace + "rights"
)

// Dublin Core Terms Properties
const (
	DCTermsCreated    = DCTermsNamespace + "created"
	DCTermsModified   = DCTermsNamespace + "modified"
	DCTermsValid      = DCTermsNamespace + "valid"
	DCTermsAvailable  = DCTermsNamespace + "available"
	DCTermsIssued     = DCTermsNamespace + "issued"
	DCTermsSubject    = DCTermsNamespace + "subject"
	DCTermsHasPart    = DCTermsNamespace + "hasPart"
	DCTermsIsPartOf   = DCTermsNamespace + "isPartOf"
	DCTermsIsVersionOf = DCTermsNamespace + "isVersionOf"
	DCTermsHasVersion = DCTermsNamespace + "hasVersion"
	DCTermsReplaces   = DCTermsNamespace + "replaces"
	DCTermsIsReplacedBy = DCTermsNamespace + "isReplacedBy"
	DCTermsReferences = DCTermsNamespace + "references"
	DCTermsIsReferencedBy = DCTermsNamespace + "isReferencedBy"
)

// SKOS Properties
const (
	SKOSPrefLabel   = SKOSNamespace + "prefLabel"
	SKOSAltLabel    = SKOSNamespace + "altLabel"
	SKOSHiddenLabel = SKOSNamespace + "hiddenLabel"
	SKOSDefinition  = SKOSNamespace + "definition"
	SKOSNote        = SKOSNamespace + "note"
	SKOSExample     = SKOSNamespace + "example"
	SKOSBroader     = SKOSNamespace + "broader"
	SKOSNarrower    = SKOSNamespace + "narrower"
	SKOSRelated     = SKOSNamespace + "related"
	SKOSExactMatch  = SKOSNamespace + "exactMatch"
	SKOSCloseMatch  = SKOSNamespace + "closeMatch"
	SKOSBroadMatch  = SKOSNamespace + "broadMatch"
	SKOSNarrowMatch = SKOSNamespace + "narrowMatch"
	SKOSRelatedMatch = SKOSNamespace + "relatedMatch"
	SKOSInScheme    = SKOSNamespace + "inScheme"
	SKOSTopConceptOf = SKOSNamespace + "topConceptOf"
	SKOSConcept      = SKOSNamespace + "Concept"
	SKOSConceptScheme = SKOSNamespace + "ConceptScheme"
	SKOSCollection   = SKOSNamespace + "Collection"
	SKOSOrderedCollection = SKOSNamespace + "OrderedCollection"
)

